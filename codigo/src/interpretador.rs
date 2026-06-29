use crate::ast::*;
use mysql::prelude::*;
/// Interpretador da linguagem PEP  -  executa a AST
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// Crates externas usadas nas funcoes nativas
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

// -- Thread-locals de rastreamento de erros ------------------------------------
// Usados por lancar/capturar para transmitir Valor::Erro rico sem mudar
// todas as assinaturas de Result<T, String>.

thread_local! {
    static LINHA_ATUAL: Cell<usize> = const { Cell::new(0) };
    static ERRO_USUARIO: RefCell<Option<Valor>> = RefCell::new(None);
}

// -- Buffers thread-local para modo servidor -----------------------------------
// Cada worker thread tem seu proprio contexto HTTP, sem compartilhamento.

thread_local! {
    static MODO_SERVIDOR: Cell<bool> = const { Cell::new(false) };
    static SAIDA_HTTP:    RefCell<Vec<u8>>                  = RefCell::new(Vec::new());
    static STATUS_HTTP:   Cell<u16>                         = const { Cell::new(200) };
    static HDRS_HTTP:     RefCell<Vec<(String, String)>>    = RefCell::new(Vec::new());
}

pub fn iniciar_contexto_servidor() {
    MODO_SERVIDOR.with(|m| m.set(true));
    SAIDA_HTTP.with(|s| s.borrow_mut().clear());
    STATUS_HTTP.with(|s| s.set(200));
    HDRS_HTTP.with(|h| h.borrow_mut().clear());
}

// Setters públicos — usados pelos nativos da VM (resposta, definir_status, etc.)
pub fn http_definir_status(status: u16) {
    STATUS_HTTP.with(|s| s.set(status));
}
pub fn http_definir_cabecalho(k: String, v: String) {
    HDRS_HTTP.with(|h| h.borrow_mut().push((k, v)));
}
pub fn http_escrever_corpo(dados: &[u8]) {
    SAIDA_HTTP.with(|s| s.borrow_mut().extend_from_slice(dados));
}
pub fn http_limpar_corpo() {
    SAIDA_HTTP.with(|s| s.borrow_mut().clear());
}

pub fn drenar_resposta() -> (u16, Vec<(String, String)>, Vec<u8>) {
    MODO_SERVIDOR.with(|m| m.set(false));
    let status = STATUS_HTTP.with(|s| s.get());
    let hdrs = HDRS_HTTP.with(|h| h.borrow_mut().drain(..).collect());
    let corpo = SAIDA_HTTP.with(|s| s.borrow_mut().drain(..).collect());
    (status, hdrs, corpo)
}

#[inline]
fn saida_escrever(texto: &str) {
    MODO_SERVIDOR.with(|m| {
        if m.get() {
            SAIDA_HTTP.with(|s| s.borrow_mut().extend_from_slice(texto.as_bytes()));
        } else {
            print!("{}", texto);
        }
    });
}

pub(crate) fn saida_vm_escrever(texto: &str) {
    saida_escrever(texto);
}

pub(crate) fn saida_vm_flush() {
    saida_flush();
}

#[inline]
fn saida_flush() {
    if !MODO_SERVIDOR.with(|m| m.get()) {
        use std::io::Write as _;
        let _ = std::io::stdout().flush();
    }
}

// -- Tipos de valor ------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Valor {
    Inteiro(i64),
    Numero(f64),
    Texto(String),
    Booleano(bool),
    Nulo,
    Lista(Vec<Valor>),
    Mapa(HashMap<String, Valor>),
    Funcao {
        parametros: Vec<Parametro>,
        corpo: Vec<Instrucao>,
        ambiente: Ambiente,
    },
    FuncaoNativa(String),
    ConexaoBD(u64),
    ConexaoSQLite(u64),
    Erro {
        tipo: String,
        mensagem: String,
        pilha: Vec<String>,
    },
    /// Tensor n-dimensional. Para matrizes 2D: shape=[linhas, colunas].
    /// Arc permite Clone O(1) — essencial para IA com grandes tensores.
    Tensor {
        shape: Vec<usize>,
        dados: Arc<Vec<f64>>,
    },
    /// Sequência de bytes brutos (corpo HTTP, arquivos binários, etc.).
    /// Arc para Clone O(1) em payloads grandes.
    Bytes(Arc<Vec<u8>>),
}

impl fmt::Display for Valor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Valor::Inteiro(n) => write!(f, "{}", n),
            Valor::Numero(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            Valor::Texto(s) => write!(f, "{}", s),
            Valor::Booleano(b) => write!(f, "{}", if *b { "verdadeiro" } else { "falso" }),
            Valor::Nulo => write!(f, "nulo"),
            Valor::Lista(v) => {
                let s: Vec<String> = v.iter().map(|x| x.repr()).collect();
                write!(f, "[{}]", s.join(", "))
            }
            Valor::Mapa(m) => {
                let mut pares: Vec<String> = m
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.repr()))
                    .collect();
                pares.sort();
                write!(f, "{{{}}}", pares.join(", "))
            }
            Valor::Funcao { .. } => write!(f, "<funcao>"),
            Valor::FuncaoNativa(n) => write!(f, "<funcao nativa: {}>", n),
            Valor::ConexaoBD(id) => write!(f, "<conexao BD #{}>", id),
            Valor::ConexaoSQLite(id) => write!(f, "<conexao SQLite #{}>", id),
            Valor::Erro {
                tipo,
                mensagem,
                pilha,
            } => {
                write!(f, "[Erro:{}] {}", tipo, mensagem)?;
                if !pilha.is_empty() {
                    for frame in pilha {
                        write!(f, "\n  em {}", frame)?;
                    }
                }
                Ok(())
            }
            Valor::Tensor { shape, dados } => {
                write!(f, "{}", tensor_fmt(shape, dados, 0, &mut 0))
            }
            Valor::Bytes(b) => write!(f, "<bytes:{}>", b.len()),
        }
    }
}

impl Valor {
    fn repr(&self) -> String {
        match self {
            Valor::Texto(s) => format!("\"{}\"", s),
            other => other.to_string(),
        }
    }
}

impl PartialEq for Valor {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Valor::Inteiro(a), Valor::Inteiro(b)) => a == b,
            (Valor::Inteiro(a), Valor::Numero(b)) | (Valor::Numero(b), Valor::Inteiro(a)) => {
                *a as f64 == *b
            }
            (Valor::Numero(a), Valor::Numero(b)) => a == b,
            (Valor::Texto(a), Valor::Texto(b)) => a == b,
            (Valor::Booleano(a), Valor::Booleano(b)) => a == b,
            (Valor::Nulo, Valor::Nulo) => true,
            (Valor::ConexaoBD(a), Valor::ConexaoBD(b)) => a == b,
            (Valor::ConexaoSQLite(a), Valor::ConexaoSQLite(b)) => a == b,
            (
                Valor::Erro {
                    tipo: ta,
                    mensagem: ma,
                    ..
                },
                Valor::Erro {
                    tipo: tb,
                    mensagem: mb,
                    ..
                },
            ) => ta == tb && ma == mb,
            (
                Valor::Tensor {
                    shape: sa,
                    dados: da,
                },
                Valor::Tensor {
                    shape: sb,
                    dados: db,
                },
            ) => sa == sb && da == db,
            (Valor::Lista(a), Valor::Lista(b)) => a == b,
            (Valor::Mapa(a), Valor::Mapa(b)) => {
                a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).map_or(false, |bv| v == bv))
            }
            (Valor::Bytes(a), Valor::Bytes(b)) => a == b,
            _ => false,
        }
    }
}

// -- Ambiente de variaveis -----------------------------------------------------

#[derive(Debug, Clone)]
pub struct Ambiente {
    variaveis: HashMap<String, Valor>,
    pai: Option<Box<Ambiente>>,
}

impl Ambiente {
    pub fn vazio() -> Self {
        Ambiente {
            variaveis: HashMap::new(),
            pai: None,
        }
    }

    pub fn novo() -> Self {
        let mut env = Ambiente::vazio();
        // E/S
        for nome in &["imprimir", "escrever", "nova_linha", "entrada"] {
            env.reg(nome);
        }
        // Conversoes
        for nome in &["texto", "numero", "inteiro", "tipo", "booleano"] {
            env.reg(nome);
        }
        // Listas
        for nome in &[
            "tamanho",
            "adicionar",
            "remover",
            "intervalo",
            "ordenar",
            "inverter_lista",
            "contem",
            "indice_de",
            "fatiar",
            "mapear",
            "filtrar",
            "reduzir",
        ] {
            env.reg(nome);
        }
        // Mapas
        for nome in &[
            "mapa",
            "mapa_definir",
            "mapa_obter",
            "mapa_tem",
            "mapa_remover",
            "mapa_chaves",
            "mapa_valores",
            "mapa_para_lista",
        ] {
            env.reg(nome);
        }
        // Texto
        for nome in &[
            "maiusculas",
            "minusculas",
            "aparar",
            "aparar_esquerda",
            "aparar_direita",
            "substituir",
            "dividir",
            "juntar",
            "comeca_com",
            "começa_com",
            "termina_com",
            "contem_texto",
            "posicao",
            "sub_texto",
            "inverter_texto",
            "repetir",
            "html_escapar",
            "url_codificar",
            "url_decodificar_texto",
        ] {
            env.reg(nome);
        }
        // Matematica
        for nome in &[
            "raiz",
            "potencia",
            "absoluto",
            "arredondar",
            "piso",
            "teto",
            "minimo",
            "maximo",
            "aleatorio",
            "aleatorio_inteiro",
            "seno",
            "cosseno",
            "tangente",
            "logaritmo",
            "pi",
            "infinito",
            "eh_numero",
            "truncar",
        ] {
            env.reg(nome);
        }
        // Arquivos
        for nome in &[
            "ler_arquivo",
            "escrever_arquivo",
            "escrever_arquivo_bytes",
            "acrescentar_arquivo",
            "arquivo_existe",
            "apagar_arquivo",
            "listar_arquivos",
            "criar_diretorio",
            "eh_arquivo",
            "eh_diretorio",
        ] {
            env.reg(nome);
        }
        // CSV
        for nome in &[
            "csv_ler",
            "csv_ler_mapa",
            "csv_escrever",
            "csv_parsear",
            "csv_serializar",
        ] {
            env.reg(nome);
        }
        // JSON
        for nome in &["json_serializar", "json_deserializar"] {
            env.reg(nome);
        }
        // Data/hora
        for nome in &[
            "data_hora",
            "timestamp",
            "dormir",
            "formatar_numero",
            "formatar_data",
        ] {
            env.reg(nome);
        }
        // Sessao/cookie (web)
        for nome in &[
            "sessao_iniciar",
            "sessao_obter",
            "sessao_definir",
            "sessao_remover",
            "sessao_destruir",
            "sessao_renovar",
            "sessao_regenerar",
            "sessao_listar_chaves",
            "sessao_obter_tudo",
            "csrf_token",
            "csrf_verificar",
            "cookie_obter",
            "cookie_definir",
        ] {
            env.reg(nome);
        }
        // Roteamento
        env.reg("rota");
        // HTTP / Web
        for nome in &[
            "cabecalho",
            "status",
            "redirecionar",
            "json_responder",
            "entrada_get",
            "entrada_post",
            "obter",
            "obter_url",
            "postar_url",
        ] {
            env.reg(nome);
        }
        // Erros tipados
        for nome in &["Erro", "tipo_erro", "mensagem_erro", "pilha_erro"] {
            env.reg(nome);
        }
        // Vetores e Tensores (algebra linear + IA)
        for nome in &[
            // Vetores (sobre listas de numeros)
            "vec_soma",
            "vec_sub",
            "vec_mul",
            "vec_div",
            "produto_interno",
            "norma",
            "normalizar",
            "produto_vetorial",
            // Matrizes / Tensores (ndarray interno)
            "matriz",
            "mat_de",
            "mat_obter",
            "mat_definir",
            "mat_soma",
            "mat_sub",
            "mat_mul",
            "mat_transpor",
            "mat_identidade",
            "mat_linhas",
            "mat_colunas",
            "mat_para_lista",
            // API Tensor n-dimensional
            "tensor",
            "tensor_de",
            "tensor_zeros",
            "tensor_uns",
            "tensor_shape",
            "tensor_ndim",
            "tensor_tamanho",
            "tensor_reshape",
            "tensor_transpor",
            "tensor_para_lista",
            "tensor_soma",
            "tensor_sub",
            "tensor_mul",
            "tensor_div",
            "tensor_potencia",
            "tensor_neg",
            "tensor_exp",
            "tensor_log",
            "tensor_raiz",
            "tensor_relu",
            "tensor_sigmoid",
            "tensor_tanh",
            "tensor_softmax",
            "tensor_media",
            "tensor_soma_total",
            "tensor_max",
            "tensor_min",
            "tensor_soma_eixo",
            "tensor_media_eixo",
            "tensor_concatenar",
            "tensor_matmul",
            // Quantização
            "tensor_quantizar_int8",
            "tensor_dequantizar_int8",
            "tensor_quantizar_f16",
            "tensor_dequantizar_f16",
            // Imagens
            "imagem_ler",
            "imagem_ler_tensor",
            "imagem_tensor_para_rgb",
            "imagem_salvar",
            "imagem_largura",
            "imagem_altura",
            "imagem_info",
        ] {
            env.reg(nome);
        }
        // PDF
        for nome in &[
            "pdf_informacoes",
            "pdf_numero_paginas",
            "pdf_extrair_texto",
            "pdf_extrair_pagina",
            "pdf_extrair_paginas",
            "pdf_ocr_disponivel",
            "pdf_ocr_texto",
            "pdf_ocr_paginas",
            "pdf_extrair_texto_com_ocr",
        ] {
            env.reg(nome);
        }
        // FFI dinamica (desativada por padrao)
        for nome in &["ffi_permitida", "ffi_carregar", "ffi_chamar", "ffi_fechar"] {
            env.reg(nome);
        }
        // Bytes
        for nome in &[
            "bytes_de_texto",
            "bytes_para_texto",
            "bytes_tamanho",
            "bytes_fatia",
            "bytes_de_lista",
            "bytes_para_lista",
            "bytes_base64",
            "bytes_para_hex",
            "bytes_concatenar",
            "bytes_obter",
            "corpo_bruto",
        ] {
            env.reg(nome);
        }
        // Modelos globais (IA)
        for nome in &[
            "modelo_carregar",
            "modelo_obter",
            "modelo_existe",
            "modelo_listar",
            "modelo_descarregar",
        ] {
            env.reg(nome);
        }
        // WebSocket
        for nome in &[
            "ws_aceitar",
            "ws_receber",
            "ws_enviar",
            "ws_enviar_bytes",
            "ws_fechar",
            "ws_id",
            "ws_conexoes",
            "ws_enviar_para",
            "ws_broadcast",
        ] {
            env.reg(nome);
        }
        // mmap
        for nome in &[
            "mmap_abrir",
            "mmap_fechar",
            "mmap_tamanho",
            "mmap_ler_f32",
            "mmap_ler_f64",
            "mmap_tensor_f32",
            "mmap_tensor_f64",
            "mmap_ler_bytes",
        ] {
            env.reg(nome);
        }
        // Strings avancadas, buffer, regex, hash, base64
        for nome in &[
            "capturar_saida",
            "formatar",
            "regex_combinar",
            "regex_combinar_tudo",
            "regex_substituir",
            "regex_dividir",
            "base64_codificar",
            "base64_decodificar",
            "sha256",
            "md5",
            "hmac_sha256",
        ] {
            env.reg(nome);
        }
        // Banco de dados
        for nome in &["bd_conectar", "bd_consultar", "bd_executar", "bd_fechar"] {
            env.reg(nome);
        }
        for nome in &[
            "sqlite_conectar",
            "sqlite_consultar",
            "sqlite_executar",
            "sqlite_fechar",
        ] {
            env.reg(nome);
        }
        env
    }

    fn reg(&mut self, nome: &str) {
        self.variaveis
            .insert(nome.to_string(), Valor::FuncaoNativa(nome.to_string()));
    }

    pub fn filho(pai: Ambiente) -> Self {
        Ambiente {
            variaveis: HashMap::new(),
            pai: Some(Box::new(pai)),
        }
    }

    pub fn obter(&self, nome: &str) -> Option<Valor> {
        if let Some(v) = self.variaveis.get(nome) {
            Some(v.clone())
        } else if let Some(pai) = &self.pai {
            pai.obter(nome)
        } else {
            None
        }
    }

    pub fn definir(&mut self, nome: String, valor: Valor) {
        self.variaveis.insert(nome, valor);
    }

    pub fn atribuir(&mut self, nome: &str, valor: Valor) -> bool {
        if self.variaveis.contains_key(nome) {
            self.variaveis.insert(nome.to_string(), valor);
            true
        } else if let Some(pai) = &mut self.pai {
            pai.atribuir(nome, valor)
        } else {
            false
        }
    }

    fn exportar_publicas(&self) -> HashMap<String, Valor> {
        self.variaveis
            .iter()
            .filter(|(nome, valor)| {
                !nome.starts_with('_') && !matches!(valor, Valor::FuncaoNativa(_))
            })
            .map(|(nome, valor)| (nome.clone(), valor.clone()))
            .collect()
    }

    fn capturar_funcoes(&mut self) {
        let snapshot = self.clone();
        for valor in self.variaveis.values_mut() {
            if let Valor::Funcao { ambiente, .. } = valor {
                *ambiente = snapshot.clone();
            }
        }
    }
}

// -- Sinais de controle de fluxo -----------------------------------------------

enum Sinal {
    Retornar(Valor),
    Pare,
    Continue,
}

// -- Interpretador -------------------------------------------------------------

const LIMITE_OPS_PADRAO: u64 = 10_000_000;

pub struct Interpretador {
    pub ambiente: Ambiente,
    conexoes_bd: RefCell<HashMap<u64, mysql::Conn>>,
    conexoes_sqlite: RefCell<HashMap<u64, rusqlite::Connection>>,
    proximo_id_bd: Cell<u64>,
    modulos_importados: RefCell<HashMap<String, HashMap<String, Valor>>>,
    arquivos_incluidos: RefCell<HashSet<String>>,
    diretorios_importacao: RefCell<Vec<PathBuf>>,
    contador_ops: Cell<u64>,
    limite_ops: u64,
    pilha_chamadas: RefCell<Vec<String>>,
}

impl Interpretador {
    pub fn novo() -> Self {
        let limite = std::env::var("PEP_MAX_OPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(LIMITE_OPS_PADRAO);
        Interpretador {
            ambiente: Ambiente::novo(),
            conexoes_bd: RefCell::new(HashMap::new()),
            conexoes_sqlite: RefCell::new(HashMap::new()),
            proximo_id_bd: Cell::new(1),
            modulos_importados: RefCell::new(HashMap::new()),
            arquivos_incluidos: RefCell::new(HashSet::new()),
            diretorios_importacao: RefCell::new(Vec::new()),
            contador_ops: Cell::new(0),
            limite_ops: limite,
            pilha_chamadas: RefCell::new(Vec::new()),
        }
    }

    fn capturar_pilha(&self) -> Vec<String> {
        let linha_erro = LINHA_ATUAL.with(|l| l.get());
        let frames = self.pilha_chamadas.borrow();
        if frames.is_empty() {
            return vec![];
        }

        // Cada frame foi registrado com a linha do CALL SITE (linha no caller).
        // Para o display queremos:
        //   frame[innermost] -> linha onde o erro ocorreu (linha_erro)
        //   frame[outer]     -> linha onde o frame mais interno foi chamado (= linha do frame seguinte)
        let parsed: Vec<(&str, usize)> = frames
            .iter()
            .map(|f| {
                if let Some(pos) = f.rfind(" (linha ") {
                    let nome = &f[..pos];
                    let linha: usize = f[pos + 8..f.len() - 1].parse().unwrap_or(0);
                    (nome, linha)
                } else {
                    (f.as_str(), 0)
                }
            })
            .collect();

        let n = parsed.len();
        let mut resultado = Vec::with_capacity(n);
        for i in (0..n).rev() {
            let (nome, _) = parsed[i];
            let linha = if i == n - 1 {
                linha_erro
            } else {
                parsed[i + 1].1
            };
            if linha > 0 {
                resultado.push(format!("{} (linha {})", nome, linha));
            } else {
                resultado.push(nome.to_string());
            }
        }
        resultado
    }

    pub fn executar(&mut self, programa: &Programa) -> Result<(), String> {
        self.executar_com_retorno(programa).map(|_| ())
    }

    /// Executa um programa e preserva um `retornar` no nivel superior.
    pub fn executar_com_retorno(&mut self, programa: &Programa) -> Result<Valor, String> {
        let mut env = self.ambiente.clone();
        let resultado = self.executar_bloco(programa, &mut env);
        self.ambiente = env;
        resultado.map(|sinal| match sinal {
            Some(Sinal::Retornar(valor)) => valor,
            _ => Valor::Nulo,
        })
    }

    /// Chama um Valor::Funcao (ou nativa) a partir de codigo externo (ex: servidor de rotas).
    pub fn chamar_valor_pub(
        &self,
        nome: &str,
        funcao: Valor,
        args: Vec<Valor>,
    ) -> Result<Valor, String> {
        let mut env = self.ambiente.clone();
        self.chamar_valor(nome, funcao, args, &mut env)
    }

    /// Ponte de nativas para a VM. Executa apenas a biblioteca nativa; nenhum
    /// AST e avaliado por este caminho.
    pub(crate) fn chamar_nativa_vm(
        &self,
        nome: &str,
        args: Vec<Valor>,
        variaveis: HashMap<String, Valor>,
    ) -> Result<Valor, String> {
        let mut env = Ambiente::novo();
        for (chave, valor) in variaveis {
            env.definir(chave, valor);
        }
        self.chamar_nativa(nome, args, &mut env)
    }

    pub fn entrar_diretorio_importacao(&self, dir: PathBuf) {
        self.diretorios_importacao.borrow_mut().push(dir);
    }

    pub fn sair_diretorio_importacao(&self) {
        self.diretorios_importacao.borrow_mut().pop();
    }

    fn executar_bloco(
        &self,
        instrucoes: &[Instrucao],
        env: &mut Ambiente,
    ) -> Result<Option<Sinal>, String> {
        for instrucao in instrucoes {
            if let Some(sinal) = self.executar_instrucao(instrucao, env)? {
                return Ok(Some(sinal));
            }
        }
        Ok(None)
    }

    fn executar_instrucao(
        &self,
        instrucao: &Instrucao,
        env: &mut Ambiente,
    ) -> Result<Option<Sinal>, String> {
        match instrucao {
            Instrucao::Localizada {
                linha,
                contexto,
                instrucao,
            } => {
                LINHA_ATUAL.with(|l| l.set(*linha));
                return self.executar_instrucao(instrucao, env).map_err(|erro| {
                    if erro.starts_with("Erro na linha ") {
                        erro
                    } else {
                        format!(
                            "Erro na linha {}: {}\n  -> {}",
                            linha,
                            erro,
                            contexto.trim()
                        )
                    }
                });
            }

            Instrucao::Expressao(expr) => {
                self.avaliar(expr, env)?;
                Ok(None)
            }

            Instrucao::DeclararVar { nome, valor } => {
                let v = match valor {
                    Some(expr) => self.avaliar(expr, env)?,
                    None => Valor::Nulo,
                };
                env.definir(nome.clone(), v);
                Ok(None)
            }

            Instrucao::Imprimir(args) => {
                let mut partes = Vec::new();
                for a in args {
                    partes.push(self.avaliar(a, env)?.to_string());
                }
                saida_escrever(&partes.join(" "));
                saida_escrever("\n");
                saida_flush();
                Ok(None)
            }

            Instrucao::Se {
                condicao,
                entao,
                senao,
            } => {
                let cond = self.avaliar(condicao, env)?;
                if self.e_verdadeiro(&cond) {
                    self.executar_bloco(entao, env)
                } else if let Some(bloco) = senao {
                    self.executar_bloco(bloco, env)
                } else {
                    Ok(None)
                }
            }

            Instrucao::Enquanto { condicao, corpo } => {
                loop {
                    let ops = self.contador_ops.get() + 1;
                    self.contador_ops.set(ops);
                    if self.limite_ops > 0 && ops > self.limite_ops {
                        return Err(format!(
                            "Limite de {} operacoes excedido. Possivel loop infinito. Use PEP_MAX_OPS para ajustar.",
                            self.limite_ops
                        ));
                    }
                    let cond = self.avaliar(condicao, env)?;
                    if !self.e_verdadeiro(&cond) {
                        break;
                    }
                    match self.executar_bloco(corpo, env)? {
                        Some(Sinal::Pare) => break,
                        Some(Sinal::Continue) => continue,
                        Some(s @ Sinal::Retornar(_)) => return Ok(Some(s)),
                        None => {}
                    }
                }
                Ok(None)
            }

            Instrucao::Para {
                variavel,
                iteravel,
                corpo,
            } => {
                let iter_val = self.avaliar(iteravel, env)?;
                let itens = match iter_val {
                    Valor::Lista(v) => v,
                    Valor::Texto(s) => s.chars().map(|c| Valor::Texto(c.to_string())).collect(),
                    _ => return Err("'para...em' requer uma lista ou texto".to_string()),
                };
                for item in itens {
                    env.definir(variavel.clone(), item);
                    match self.executar_bloco(corpo, env)? {
                        Some(Sinal::Pare) => break,
                        Some(Sinal::Continue) => continue,
                        Some(s @ Sinal::Retornar(_)) => return Ok(Some(s)),
                        None => {}
                    }
                }
                Ok(None)
            }

            Instrucao::ParaIntervalo {
                variavel,
                inicio,
                fim,
                passo,
                corpo,
            } => {
                let vi = self.avaliar(inicio, env)?;
                let vf = self.avaliar(fim, env)?;
                let vp = match passo {
                    Some(p) => self.avaliar(p, env)?,
                    None => Valor::Inteiro(1),
                };
                let (start, end, step) = match (&vi, &vf, &vp) {
                    (Valor::Inteiro(s), Valor::Inteiro(e), Valor::Inteiro(p)) => (*s, *e, *p),
                    (Valor::Numero(s), Valor::Numero(e), Valor::Numero(p)) => {
                        (*s as i64, *e as i64, *p as i64)
                    }
                    (Valor::Inteiro(s), Valor::Numero(e), Valor::Inteiro(p)) => (*s, *e as i64, *p),
                    (Valor::Numero(s), Valor::Inteiro(e), Valor::Inteiro(p)) => (*s as i64, *e, *p),
                    _ => return Err("'para de ate' requer numeros inteiros".to_string()),
                };
                if step == 0 {
                    return Err("passo nao pode ser zero".to_string());
                }
                let mut i = start;
                loop {
                    let ops = self.contador_ops.get() + 1;
                    self.contador_ops.set(ops);
                    if self.limite_ops > 0 && ops > self.limite_ops {
                        return Err(format!("Limite de {} operacoes excedido.", self.limite_ops));
                    }
                    let continuar = if step > 0 { i <= end } else { i >= end };
                    if !continuar {
                        break;
                    }
                    env.definir(variavel.clone(), Valor::Inteiro(i));
                    match self.executar_bloco(corpo, env)? {
                        Some(Sinal::Pare) => break,
                        Some(Sinal::Continue) => {}
                        Some(s @ Sinal::Retornar(_)) => return Ok(Some(s)),
                        None => {}
                    }
                    i = i.saturating_add(step);
                }
                Ok(None)
            }

            Instrucao::Escolher {
                expr,
                casos,
                padrao,
            } => {
                let valor = self.avaliar(expr, env)?;
                for (valores_caso, bloco) in casos {
                    for v_expr in valores_caso {
                        let v = self.avaliar(v_expr, env)?;
                        if valor == v {
                            return self.executar_bloco(bloco, env);
                        }
                    }
                }
                if let Some(bloco) = padrao {
                    return self.executar_bloco(bloco, env);
                }
                Ok(None)
            }

            Instrucao::Funcao {
                nome,
                parametros,
                corpo,
            } => {
                let funcao = Valor::Funcao {
                    parametros: parametros.clone(),
                    corpo: corpo.clone(),
                    ambiente: env.clone(),
                };
                env.definir(nome.clone(), funcao);
                Ok(None)
            }

            Instrucao::Retornar(expr) => {
                let v = match expr {
                    Some(e) => self.avaliar(e, env)?,
                    None => Valor::Nulo,
                };
                Ok(Some(Sinal::Retornar(v)))
            }

            Instrucao::Pare => Ok(Some(Sinal::Pare)),
            Instrucao::Continue => Ok(Some(Sinal::Continue)),

            // -- Tratamento de erros -------------------------------------------
            Instrucao::Lancar(expr) => {
                let val = self.avaliar(expr, env)?;
                let erro = match val {
                    Valor::Erro { tipo, mensagem, .. } => Valor::Erro {
                        pilha: self.capturar_pilha(),
                        tipo,
                        mensagem,
                    },
                    outro => Valor::Erro {
                        tipo: "Erro".to_string(),
                        mensagem: outro.to_string(),
                        pilha: self.capturar_pilha(),
                    },
                };
                let msg = match &erro {
                    Valor::Erro { mensagem, .. } => mensagem.clone(),
                    _ => unreachable!(),
                };
                ERRO_USUARIO.with(|e| *e.borrow_mut() = Some(erro));
                Err(msg)
            }

            Instrucao::Tentar {
                corpo,
                capturar,
                finalmente,
            } => {
                let resultado = self.executar_bloco(corpo, env);
                let sinal = match resultado {
                    Ok(s) => {
                        // limpa qualquer erro pendente que nao foi relancado
                        ERRO_USUARIO.with(|e| *e.borrow_mut() = None);
                        Ok(s)
                    }
                    Err(msg) => {
                        // extrai Valor::Erro rico se disponivel, senao cria basico
                        let val_erro =
                            ERRO_USUARIO
                                .with(|e| e.borrow_mut().take())
                                .unwrap_or_else(|| {
                                    let linha = LINHA_ATUAL.with(|l| l.get());
                                    // extrai mensagem limpa descartando "Erro na linha X: "
                                    let mensagem = if let Some(pos) = msg.find('\n') {
                                        msg[..pos].to_string()
                                    } else {
                                        msg.clone()
                                    };
                                    Valor::Erro {
                                        tipo: "Erro".to_string(),
                                        mensagem,
                                        pilha: if linha > 0 {
                                            vec![format!("(linha {})", linha)]
                                        } else {
                                            vec![]
                                        },
                                    }
                                });
                        if let Some((nome_erro, bloco)) = capturar {
                            env.definir(nome_erro.clone(), val_erro);
                            self.executar_bloco(bloco, env)
                        } else {
                            // relanca: restaura o Valor::Erro no thread-local
                            ERRO_USUARIO.with(|e| *e.borrow_mut() = Some(val_erro));
                            Err(msg)
                        }
                    }
                };
                // finalmente sempre executa
                if let Some(fin) = finalmente {
                    let _ = self.executar_bloco(fin, env);
                }
                sinal
            }

            // -- Importacao de modulos -----------------------------------------
            Instrucao::Importar { caminho, alias } => {
                let exports = self.importar_modulo(caminho)?;
                if let Some(nome) = alias {
                    env.definir(nome.clone(), Valor::Mapa(exports));
                } else {
                    for (nome, valor) in exports {
                        env.definir(nome, valor);
                    }
                }
                Ok(None)
            }

            Instrucao::Incluir {
                caminho,
                obrigatorio,
            } => {
                self.incluir_arquivo(caminho, *obrigatorio, env)?;
                Ok(None)
            }
        }
    }

    fn incluir_arquivo(
        &self,
        caminho: &str,
        obrigatorio: bool,
        env: &mut Ambiente,
    ) -> Result<(), String> {
        let caminho_resolvido = self.resolver_caminho_import(caminho);
        if !caminho_resolvido.exists() {
            if obrigatorio {
                return Err(format!("requerer '{}': arquivo nao encontrado", caminho));
            }
            return Ok(());
        }

        let caminho_abs = std::fs::canonicalize(&caminho_resolvido)
            .map_err(|e| format!("incluir '{}': {}", caminho, e))?;
        let chave = caminho_abs.to_string_lossy().to_string();
        if !self.arquivos_incluidos.borrow_mut().insert(chave) {
            return Ok(());
        }
        let fonte = std::fs::read_to_string(&caminho_abs)
            .map_err(|e| format!("incluir '{}': {}", caminho_abs.display(), e))?;

        let ext = caminho_abs
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let prog = if matches!(ext, "phtml" | "html" | "htm") {
            crate::template::compilar(&fonte)?
        } else {
            let mut lex = crate::lexer::Lexer::novo(&fonte);
            let tokens = lex.tokenizar()?;
            let mut p = crate::parser::Parser::novo(tokens);
            p.parsear()?
        };

        if let Some(dir) = caminho_abs.parent() {
            self.diretorios_importacao
                .borrow_mut()
                .push(dir.to_path_buf());
        }
        let resultado = self.executar_bloco(&prog, env);
        if caminho_abs.parent().is_some() {
            self.diretorios_importacao.borrow_mut().pop();
        }
        resultado.map(|_| ())
    }

    fn importar_modulo(&self, caminho: &str) -> Result<HashMap<String, Valor>, String> {
        let caminho_resolvido = self.resolver_caminho_import(caminho);
        let caminho_abs = std::fs::canonicalize(&caminho_resolvido)
            .map_err(|e| format!("importar '{}': {}", caminho, e))?;
        let chave = caminho_abs.to_string_lossy().to_string();

        if let Some(exports) = self.modulos_importados.borrow().get(&chave).cloned() {
            return Ok(exports);
        }

        let fonte = std::fs::read_to_string(&caminho_abs)
            .map_err(|e| format!("importar '{}': {}", caminho_abs.display(), e))?;

        let mut lex = crate::lexer::Lexer::novo(&fonte);
        let tokens = lex.tokenizar()?;
        let mut p = crate::parser::Parser::novo(tokens);
        let prog = p.parsear()?;

        let mut env_modulo = Ambiente::novo();
        if let Some(dir) = caminho_abs.parent() {
            self.diretorios_importacao
                .borrow_mut()
                .push(dir.to_path_buf());
        }
        let resultado = self.executar_bloco(&prog, &mut env_modulo);
        if caminho_abs.parent().is_some() {
            self.diretorios_importacao.borrow_mut().pop();
        }
        resultado?;

        env_modulo.capturar_funcoes();
        let exports = env_modulo.exportar_publicas();
        self.modulos_importados
            .borrow_mut()
            .insert(chave, exports.clone());
        Ok(exports)
    }

    fn resolver_caminho_import(&self, caminho: &str) -> PathBuf {
        let p = Path::new(caminho);
        if p.is_absolute() {
            return p.to_path_buf();
        }
        let base = self
            .diretorios_importacao
            .borrow()
            .last()
            .cloned()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_default();

        if let Some(encontrado) = localizar_modulo(base.join(p)) {
            return encontrado;
        }

        for ancestral in base.ancestors() {
            if let Some(encontrado) = localizar_modulo(ancestral.join("pep_modules").join(p)) {
                return encontrado;
            }
        }

        base.join(p)
    }

    fn avaliar(&self, expr: &Expressao, env: &mut Ambiente) -> Result<Valor, String> {
        match expr {
            Expressao::Inteiro(n) => Ok(Valor::Inteiro(*n)),
            Expressao::Numero(n) => Ok(Valor::Numero(*n)),
            Expressao::Texto(s) => Ok(Valor::Texto(s.clone())),
            Expressao::Booleano(b) => Ok(Valor::Booleano(*b)),
            Expressao::Nulo => Ok(Valor::Nulo),

            Expressao::FuncaoAnonima { parametros, corpo } => Ok(Valor::Funcao {
                parametros: parametros.clone(),
                corpo: corpo.clone(),
                ambiente: env.clone(),
            }),

            Expressao::Lista(elementos) => {
                let vals: Result<Vec<Valor>, String> =
                    elementos.iter().map(|e| self.avaliar(e, env)).collect();
                Ok(Valor::Lista(vals?))
            }

            Expressao::Mapa(pares) => {
                let mut m = HashMap::new();
                for (k, v) in pares {
                    m.insert(k.clone(), self.avaliar(v, env)?);
                }
                Ok(Valor::Mapa(m))
            }

            Expressao::Variavel(nome) => env
                .obter(nome)
                .ok_or_else(|| format!("Variavel '{}' nao definida", nome)),

            Expressao::Atribuicao { nome, valor } => {
                let v = self.avaliar(valor, env)?;
                if !env.atribuir(nome, v.clone()) {
                    return Err(format!(
                        "Variavel '{}' nao declarada. Use 'var {} = ...'",
                        nome, nome
                    ));
                }
                Ok(v)
            }

            Expressao::AtribuicaoIndexada {
                objeto,
                indice,
                valor,
            } => {
                let obj = self.avaliar(objeto, env)?;
                let idx = self.avaliar(indice, env)?;
                let val = self.avaliar(valor, env)?;

                let obj_novo = match (obj, &idx) {
                    (Valor::Lista(mut v), Valor::Inteiro(n)) => {
                        let i = *n as usize;
                        if i < v.len() {
                            v[i] = val;
                            Ok(Valor::Lista(v))
                        } else {
                            Err(format!(
                                "Indice {} fora dos limites (tamanho: {})",
                                i,
                                v.len()
                            ))
                        }
                    }
                    (Valor::Mapa(mut m), Valor::Texto(k)) => {
                        m.insert(k.clone(), val);
                        Ok(Valor::Mapa(m))
                    }
                    _ => Err(
                        "Atribuicao indexada invalida: use lista[n] ou mapa[\"chave\"]".to_string(),
                    ),
                }?;

                // Atualizar variavel no ambiente
                if let Expressao::Variavel(nome) = objeto.as_ref() {
                    env.atribuir(nome, obj_novo.clone());
                }
                Ok(obj_novo)
            }

            Expressao::UnOp { op, expr } => {
                let v = self.avaliar(expr, env)?;
                match op {
                    OpUnario::Negativo => match v {
                        Valor::Inteiro(n) => Ok(Valor::Inteiro(-n)),
                        Valor::Numero(n) => Ok(Valor::Numero(-n)),
                        _ => Err("O operador '-' so funciona com numeros".to_string()),
                    },
                    OpUnario::Nao => Ok(Valor::Booleano(!self.e_verdadeiro(&v))),
                }
            }

            Expressao::BinOp { esq, op, dir } => {
                // Curto-circuito para 'e' e 'ou'
                match op {
                    OpBinario::E => {
                        let ve = self.avaliar(esq, env)?;
                        if !self.e_verdadeiro(&ve) {
                            return Ok(Valor::Booleano(false));
                        }
                        let vd = self.avaliar(dir, env)?;
                        return Ok(Valor::Booleano(self.e_verdadeiro(&vd)));
                    }
                    OpBinario::Ou => {
                        let ve = self.avaliar(esq, env)?;
                        if self.e_verdadeiro(&ve) {
                            return Ok(Valor::Booleano(true));
                        }
                        let vd = self.avaliar(dir, env)?;
                        return Ok(Valor::Booleano(self.e_verdadeiro(&vd)));
                    }
                    _ => {}
                }
                let ve = self.avaliar(esq, env)?;
                let vd = self.avaliar(dir, env)?;
                self.aplicar_op(op, ve, vd)
            }

            Expressao::ChamadaFuncao { nome, args } => {
                let funcao = env
                    .obter(nome)
                    .ok_or_else(|| format!("Funcao '{}' nao definida", nome))?;
                let mut args_avaliados = Vec::new();
                for a in args {
                    args_avaliados.push(self.avaliar(a, env)?);
                }
                self.chamar_valor(nome, funcao, args_avaliados, env)
            }

            Expressao::Chamada { funcao, args } => {
                let funcao_valor = self.avaliar(funcao, env)?;
                let mut args_avaliados = Vec::new();
                for a in args {
                    args_avaliados.push(self.avaliar(a, env)?);
                }
                self.chamar_valor("<expressao>", funcao_valor, args_avaliados, env)
            }

            Expressao::Acesso { objeto, indice } => {
                let obj = self.avaliar(objeto, env)?;
                let idx = self.avaliar(indice, env)?;
                match (obj, idx) {
                    (Valor::Lista(v), Valor::Inteiro(n)) => {
                        let i = n as usize;
                        v.get(i).cloned().ok_or_else(|| {
                            format!("Indice {} fora dos limites (tamanho: {})", i, v.len())
                        })
                    }
                    (Valor::Texto(s), Valor::Inteiro(n)) => {
                        let i = n as usize;
                        s.chars()
                            .nth(i)
                            .map(|c| Valor::Texto(c.to_string()))
                            .ok_or_else(|| format!("Indice {} fora dos limites", i))
                    }
                    (Valor::Mapa(m), Valor::Texto(k)) => {
                        Ok(m.get(&k).cloned().unwrap_or(Valor::Nulo))
                    }
                    (Valor::Bytes(b), Valor::Inteiro(n)) => {
                        let i = n as usize;
                        b.get(i)
                            .map(|&x| Valor::Inteiro(x as i64))
                            .ok_or_else(|| format!("Indice {} fora dos limites (bytes)", i))
                    }
                    (obj, idx) => Err(format!("Acesso invalido: {} com indice {}", obj, idx)),
                }
            }

            Expressao::AcessoOpcional { objeto, chave } => {
                let obj = self.avaliar(objeto, env)?;
                match obj {
                    Valor::Nulo => Ok(Valor::Nulo),
                    Valor::Mapa(m) => Ok(m.get(chave).cloned().unwrap_or(Valor::Nulo)),
                    v => Err(format!("'?.' requer mapa ou nulo, recebeu {}", v)),
                }
            }

            Expressao::NullCoalescente { esq, dir } => {
                let v = self.avaliar(esq, env)?;
                match v {
                    Valor::Nulo => self.avaliar(dir, env),
                    outro => Ok(outro),
                }
            }

            Expressao::FuncaoSeta { parametros, corpo } => Ok(Valor::Funcao {
                parametros: parametros
                    .iter()
                    .map(|p| crate::ast::Parametro {
                        nome: p.clone(),
                        padrao: None,
                        variadic: false,
                    })
                    .collect(),
                corpo: vec![crate::ast::Instrucao::Retornar(Some(*corpo.clone()))],
                ambiente: env.clone(),
            }),
        }
    }

    // -- Operacoes binarias ----------------------------------------------------

    fn chamar_valor(
        &self,
        nome: &str,
        funcao: Valor,
        args_avaliados: Vec<Valor>,
        env: &mut Ambiente,
    ) -> Result<Valor, String> {
        match funcao {
            Valor::FuncaoNativa(nome_nativa) => {
                self.chamar_nativa(&nome_nativa, args_avaliados, env)
            }
            Valor::Funcao {
                parametros,
                corpo,
                ambiente,
            } => {
                let tem_variadic = parametros.last().map(|p| p.variadic).unwrap_or(false);
                let min_args = parametros
                    .iter()
                    .filter(|p| p.padrao.is_none() && !p.variadic)
                    .count();
                let max_args = if tem_variadic {
                    usize::MAX
                } else {
                    parametros.len()
                };

                if args_avaliados.len() < min_args || args_avaliados.len() > max_args {
                    let esperado = if tem_variadic {
                        format!("ao menos {}", min_args)
                    } else if min_args == parametros.len() {
                        format!("{}", min_args)
                    } else {
                        format!("{} a {}", min_args, parametros.len())
                    };
                    return Err(format!(
                        "Funcao '{}' espera {} argumento(s), recebeu {}",
                        nome,
                        esperado,
                        args_avaliados.len()
                    ));
                }

                let funcao_recursiva = Valor::Funcao {
                    parametros: parametros.clone(),
                    corpo: corpo.clone(),
                    ambiente: ambiente.clone(),
                };
                let base = if ambiente.variaveis.is_empty() && ambiente.pai.is_none() {
                    env.clone()
                } else {
                    ambiente
                };
                let mut env_fn = Ambiente::filho(base);
                // Propaga variáveis de contexto HTTP (_PARAMS, _CORPO_JSON, etc.)
                // do env do chamador para que sejam visíveis dentro do closure
                for (k, v) in &env.variaveis {
                    if k.starts_with('_') {
                        env_fn.definir(k.clone(), v.clone());
                    }
                }
                if nome != "<expressao>" {
                    env_fn.definir(nome.to_string(), funcao_recursiva);
                }
                for (i, param) in parametros.iter().enumerate() {
                    if param.variadic {
                        let restantes: Vec<Valor> = args_avaliados[i..].to_vec();
                        env_fn.definir(param.nome.clone(), Valor::Lista(restantes));
                        break;
                    } else if i < args_avaliados.len() {
                        env_fn.definir(param.nome.clone(), args_avaliados[i].clone());
                    } else {
                        let val = match &param.padrao {
                            Some(expr) => self.avaliar(expr, env)?,
                            None => Valor::Nulo,
                        };
                        env_fn.definir(param.nome.clone(), val);
                    }
                }
                let linha = LINHA_ATUAL.with(|l| l.get());
                let frame = if linha > 0 {
                    format!("{} (linha {})", nome, linha)
                } else {
                    nome.to_string()
                };
                self.pilha_chamadas.borrow_mut().push(frame);
                let resultado = self.executar_bloco(&corpo, &mut env_fn);
                self.pilha_chamadas.borrow_mut().pop();
                match resultado? {
                    Some(Sinal::Retornar(v)) => Ok(v),
                    _ => Ok(Valor::Nulo),
                }
            }
            _ => Err(format!("'{}' nao e uma funcao", nome)),
        }
    }

    fn aplicar_op(&self, op: &OpBinario, a: Valor, b: Valor) -> Result<Valor, String> {
        match op {
            OpBinario::Soma => match (a, b) {
                (Valor::Inteiro(x), Valor::Inteiro(y)) => Ok(Valor::Inteiro(x + y)),
                (Valor::Inteiro(x), Valor::Numero(y)) => Ok(Valor::Numero(x as f64 + y)),
                (Valor::Numero(x), Valor::Inteiro(y)) => Ok(Valor::Numero(x + y as f64)),
                (Valor::Numero(x), Valor::Numero(y)) => Ok(Valor::Numero(x + y)),
                (Valor::Texto(x), y) => Ok(Valor::Texto(x + &y.to_string())),
                (x, Valor::Texto(y)) => Ok(Valor::Texto(x.to_string() + &y)),
                (Valor::Lista(mut x), Valor::Lista(y)) => {
                    x.extend(y);
                    Ok(Valor::Lista(x))
                }
                (a, b) => Err(format!("Nao e possivel somar {} e {}", a, b)),
            },
            OpBinario::Subtracao => self.op_aritmetico(a, b, "-", |x, y| x - y, |x, y| x - y),
            OpBinario::Multiplicacao => match (a, b) {
                (Valor::Inteiro(x), Valor::Inteiro(y)) => Ok(Valor::Inteiro(x * y)),
                (Valor::Inteiro(x), Valor::Numero(y)) => Ok(Valor::Numero(x as f64 * y)),
                (Valor::Numero(x), Valor::Inteiro(y)) => Ok(Valor::Numero(x * y as f64)),
                (Valor::Numero(x), Valor::Numero(y)) => Ok(Valor::Numero(x * y)),
                (Valor::Texto(s), Valor::Inteiro(n)) | (Valor::Inteiro(n), Valor::Texto(s)) => {
                    Ok(Valor::Texto(s.repeat(n.max(0) as usize)))
                }
                (Valor::Texto(s), Valor::Numero(n)) | (Valor::Numero(n), Valor::Texto(s)) => {
                    Ok(Valor::Texto(s.repeat(n.max(0.0) as usize)))
                }
                (a, b) => Err(format!("Nao e possivel multiplicar {} e {}", a, b)),
            },
            OpBinario::Divisao => {
                let (x, y) = numeros_f64(a, b, "/")?;
                if y == 0.0 {
                    Err("Divisao por zero".to_string())
                } else {
                    Ok(Valor::Numero(x / y))
                }
            }
            OpBinario::DivisaoInteira => {
                let (x, y) = inteiros_i64(a, b, "//")?;
                if y == 0 {
                    Err("Divisao por zero".to_string())
                } else {
                    Ok(Valor::Inteiro(x / y))
                }
            }
            OpBinario::Modulo => self.op_aritmetico(a, b, "%", |x, y| x % y, |x, y| x % y),
            OpBinario::Igual => Ok(Valor::Booleano(a == b)),
            OpBinario::DiferenteDe => Ok(Valor::Booleano(a != b)),
            OpBinario::MenorQue => self.op_cmp(a, b, "<"),
            OpBinario::MaiorQue => self.op_cmp(a, b, ">"),
            OpBinario::MenorOuIgual => self.op_cmp(a, b, "<="),
            OpBinario::MaiorOuIgual => self.op_cmp(a, b, ">="),
            // E/Ou ja tratados em BinOp com curto-circuito
            OpBinario::E => Ok(Valor::Booleano(
                self.e_verdadeiro(&a) && self.e_verdadeiro(&b),
            )),
            OpBinario::Ou => Ok(Valor::Booleano(
                self.e_verdadeiro(&a) || self.e_verdadeiro(&b),
            )),
            OpBinario::Em | OpBinario::NaoEm => {
                let pertence = match &b {
                    Valor::Lista(lista) => lista.contains(&a),
                    Valor::Mapa(mapa) => {
                        if let Valor::Texto(k) = &a {
                            mapa.contains_key(k)
                        } else {
                            return Err("'em' com mapa requer chave de texto".to_string());
                        }
                    }
                    Valor::Texto(s) => {
                        if let Valor::Texto(sub) = &a {
                            s.contains(sub.as_str())
                        } else {
                            return Err("'em' com texto requer texto como elemento".to_string());
                        }
                    }
                    _ => {
                        return Err(format!(
                            "'em' requer lista, mapa ou texto, mas recebeu {}",
                            b
                        ))
                    }
                };
                let resultado = if matches!(op, OpBinario::NaoEm) {
                    !pertence
                } else {
                    pertence
                };
                Ok(Valor::Booleano(resultado))
            }
        }
    }

    fn op_aritmetico(
        &self,
        a: Valor,
        b: Valor,
        op: &str,
        inteiro: impl Fn(i64, i64) -> i64,
        decimal: impl Fn(f64, f64) -> f64,
    ) -> Result<Valor, String> {
        match (a, b) {
            (Valor::Inteiro(x), Valor::Inteiro(y)) => Ok(Valor::Inteiro(inteiro(x, y))),
            (a, b) => {
                let (x, y) = numeros_f64(a, b, op)?;
                Ok(Valor::Numero(decimal(x, y)))
            }
        }
    }

    fn op_cmp(&self, a: Valor, b: Valor, op: &str) -> Result<Valor, String> {
        if matches!(
            (&a, &b),
            (Valor::Inteiro(_), Valor::Inteiro(_))
                | (Valor::Inteiro(_), Valor::Numero(_))
                | (Valor::Numero(_), Valor::Inteiro(_))
                | (Valor::Numero(_), Valor::Numero(_))
        ) {
            let (x, y) = numeros_f64(a, b, op)?;
            return Ok(Valor::Booleano(match op {
                "<" => x < y,
                ">" => x > y,
                "<=" => x <= y,
                _ => x >= y,
            }));
        }
        match (&a, &b) {
            (Valor::Texto(x), Valor::Texto(y)) => Ok(Valor::Booleano(match op {
                "<" => x < y,
                ">" => x > y,
                "<=" => x <= y,
                _ => x >= y,
            })),
            _ => Err(format!("Nao e possivel comparar {} e {}", a, b)),
        }
    }

    fn e_verdadeiro(&self, v: &Valor) -> bool {
        match v {
            Valor::Booleano(b) => *b,
            Valor::Nulo => false,
            Valor::Numero(n) => *n != 0.0,
            Valor::Inteiro(n) => *n != 0,
            Valor::Texto(s) => !s.is_empty(),
            Valor::Lista(l) => !l.is_empty(),
            Valor::Mapa(m) => !m.is_empty(),
            Valor::Funcao { .. }
            | Valor::FuncaoNativa(_)
            | Valor::ConexaoBD(_)
            | Valor::ConexaoSQLite(_)
            | Valor::Tensor { .. } => true,
            Valor::Bytes(b) => !b.is_empty(),
            Valor::Erro { .. } => false,
        }
    }

    // -- Funcoes nativas -------------------------------------------------------

    fn chamar_nativa(
        &self,
        nome: &str,
        args: Vec<Valor>,
        env: &mut Ambiente,
    ) -> Result<Valor, String> {
        match nome {
            // -- E/S ----------------------------------------------------------
            "escrever" => {
                let partes: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                saida_escrever(&partes.join(" "));
                saida_flush();
                Ok(Valor::Nulo)
            }
            "nova_linha" => {
                saida_escrever("\n");
                saida_flush();
                Ok(Valor::Nulo)
            }
            "entrada" => {
                use std::io::Write as _;
                if let Some(v) = args.into_iter().next() {
                    print!("{}", v);
                }
                let _ = std::io::stdout().flush();
                let mut linha = String::new();
                std::io::stdin()
                    .read_line(&mut linha)
                    .map_err(|e| e.to_string())?;
                Ok(Valor::Texto(
                    linha
                        .trim_end_matches('\n')
                        .trim_end_matches('\r')
                        .to_string(),
                ))
            }

            // -- Conversoes ----------------------------------------------------
            "tipo" => {
                let t = match arg1(args, "tipo")? {
                    Valor::Inteiro(_) => "inteiro",
                    Valor::Numero(_) => "decimal",
                    Valor::Texto(_) => "texto",
                    Valor::Booleano(_) => "booleano",
                    Valor::Nulo => "nulo",
                    Valor::Lista(_) => "lista",
                    Valor::Mapa(_) => "mapa",
                    Valor::ConexaoBD(_) => "conexao_bd",
                    Valor::ConexaoSQLite(_) => "conexao_sqlite",
                    Valor::Funcao { .. } | Valor::FuncaoNativa(_) => "funcao",
                    Valor::Erro { .. } => "erro",
                    Valor::Tensor { .. } => "tensor",
                    Valor::Bytes(_) => "bytes",
                };
                Ok(Valor::Texto(t.to_string()))
            }
            "texto" => Ok(Valor::Texto(arg1(args, "texto")?.to_string())),
            "booleano" => Ok(Valor::Booleano(self.e_verdadeiro(&arg1(args, "booleano")?))),
            "numero" => match arg1(args, "numero")? {
                Valor::Inteiro(n) => Ok(Valor::Numero(n as f64)),
                Valor::Numero(n) => Ok(Valor::Numero(n)),
                Valor::Texto(s) => s
                    .trim()
                    .parse::<f64>()
                    .map(Valor::Numero)
                    .map_err(|_| format!("Nao e possivel converter '{}' para numero", s)),
                Valor::Booleano(b) => Ok(Valor::Numero(if b { 1.0 } else { 0.0 })),
                v => Err(format!("Nao e possivel converter '{}' para numero", v)),
            },
            "inteiro" => match arg1(args, "inteiro")? {
                Valor::Inteiro(n) => Ok(Valor::Inteiro(n)),
                Valor::Numero(n) => Ok(Valor::Inteiro(n.trunc() as i64)),
                Valor::Texto(s) => s
                    .trim()
                    .parse::<i64>()
                    .map(Valor::Inteiro)
                    .map_err(|_| format!("Nao e possivel converter '{}' para inteiro", s)),
                v => Err(format!("'inteiro' requer numero, recebeu {}", v)),
            },
            "truncar" => match arg1(args, "truncar")? {
                Valor::Inteiro(n) => Ok(Valor::Inteiro(n)),
                Valor::Numero(n) => Ok(Valor::Inteiro(n.trunc() as i64)),
                v => Err(format!("'truncar' requer numero, recebeu {}", v)),
            },

            // -- Listas --------------------------------------------------------
            "tamanho" => match arg1(args, "tamanho")? {
                Valor::Lista(l) => Ok(Valor::Inteiro(l.len() as i64)),
                Valor::Texto(s) => Ok(Valor::Inteiro(s.chars().count() as i64)),
                Valor::Mapa(m) => Ok(Valor::Inteiro(m.len() as i64)),
                v => Err(format!(
                    "'tamanho' requer lista, texto ou mapa, recebeu {}",
                    v
                )),
            },
            "mapear" => {
                let (lista, funcao) = arg2(args, "mapear")?;
                let itens = match lista {
                    Valor::Lista(v) => v,
                    v => return Err(format!("'mapear' requer lista, recebeu {}", v)),
                };
                let mut resultado = Vec::with_capacity(itens.len());
                for item in itens {
                    resultado.push(self.chamar_valor(
                        "<expressao>",
                        funcao.clone(),
                        vec![item],
                        env,
                    )?);
                }
                Ok(Valor::Lista(resultado))
            }
            "filtrar" => {
                let (lista, funcao) = arg2(args, "filtrar")?;
                let itens = match lista {
                    Valor::Lista(v) => v,
                    v => return Err(format!("'filtrar' requer lista, recebeu {}", v)),
                };
                let mut resultado = Vec::new();
                for item in itens {
                    let manter =
                        self.chamar_valor("<expressao>", funcao.clone(), vec![item.clone()], env)?;
                    if self.e_verdadeiro(&manter) {
                        resultado.push(item);
                    }
                }
                Ok(Valor::Lista(resultado))
            }
            "reduzir" => {
                let (itens, funcao, mut acumulador) = match args.as_slice() {
                    [Valor::Lista(v), f, inicial] => (v.clone(), f.clone(), inicial.clone()),
                    [Valor::Lista(v), f] if !v.is_empty() => {
                        (v[1..].to_vec(), f.clone(), v[0].clone())
                    }
                    _ => return Err("reduzir(lista, funcao, inicial?)".to_string()),
                };
                for item in itens {
                    acumulador = self.chamar_valor(
                        "<expressao>",
                        funcao.clone(),
                        vec![acumulador, item],
                        env,
                    )?;
                }
                Ok(acumulador)
            }
            "adicionar" => {
                let (lista, item) = arg2(args, "adicionar")?;
                match lista {
                    Valor::Lista(mut v) => {
                        v.push(item);
                        Ok(Valor::Lista(v))
                    }
                    v => Err(format!("'adicionar' requer uma lista, recebeu {}", v)),
                }
            }
            "remover" => {
                let (lista, idx) = arg2(args, "remover")?;
                match (lista, idx) {
                    (Valor::Lista(mut v), Valor::Inteiro(n)) => {
                        let i = n as usize;
                        if i < v.len() {
                            v.remove(i);
                            Ok(Valor::Lista(v))
                        } else {
                            Err(format!("Indice {} fora dos limites", i))
                        }
                    }
                    _ => Err("'remover' requer (lista, numero)".to_string()),
                }
            }
            "intervalo" => {
                let mut it = args.into_iter();
                match (it.next(), it.next(), it.next()) {
                    (Some(Valor::Inteiro(fim)), None, None) => {
                        Ok(Valor::Lista((0..fim).map(Valor::Inteiro).collect()))
                    }
                    (Some(Valor::Inteiro(ini)), Some(Valor::Inteiro(fim)), None) => {
                        Ok(Valor::Lista((ini..fim).map(Valor::Inteiro).collect()))
                    }
                    (
                        Some(Valor::Inteiro(ini)),
                        Some(Valor::Inteiro(fim)),
                        Some(Valor::Inteiro(passo)),
                    ) => {
                        let mut v = Vec::new();
                        let mut i = ini;
                        if passo == 0 {
                            return Err("intervalo: passo nao pode ser zero".to_string());
                        }
                        while (passo > 0 && i < fim) || (passo < 0 && i > fim) {
                            v.push(Valor::Inteiro(i));
                            i += passo;
                        }
                        Ok(Valor::Lista(v))
                    }
                    _ => Err(
                        "intervalo(fim) ou intervalo(ini, fim) ou intervalo(ini, fim, passo)"
                            .to_string(),
                    ),
                }
            }
            "ordenar" => match arg1(args, "ordenar")? {
                Valor::Lista(mut v) => {
                    v.sort_by(|a, b| match (a, b) {
                        (Valor::Inteiro(x), Valor::Inteiro(y)) => x.cmp(y),
                        (Valor::Numero(x), Valor::Numero(y)) => {
                            x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        _ => a.to_string().cmp(&b.to_string()),
                    });
                    Ok(Valor::Lista(v))
                }
                v => Err(format!("'ordenar' requer lista, recebeu {}", v)),
            },
            "inverter_lista" => match arg1(args, "inverter_lista")? {
                Valor::Lista(mut v) => {
                    v.reverse();
                    Ok(Valor::Lista(v))
                }
                v => Err(format!("'inverter_lista' requer lista, recebeu {}", v)),
            },
            "contem" => {
                let (lista, item) = arg2(args, "contem")?;
                match lista {
                    Valor::Lista(v) => Ok(Valor::Booleano(v.contains(&item))),
                    v => Err(format!("'contem' requer lista, recebeu {}", v)),
                }
            }
            "indice_de" => {
                let (lista, item) = arg2(args, "indice_de")?;
                match lista {
                    Valor::Lista(v) => {
                        let i = v.iter().position(|x| x == &item);
                        Ok(Valor::Inteiro(i.map(|n| n as i64).unwrap_or(-1)))
                    }
                    v => Err(format!("'indice_de' requer lista, recebeu {}", v)),
                }
            }
            "fatiar" => {
                let mut it = args.into_iter();
                match (it.next(), it.next(), it.next()) {
                    (
                        Some(Valor::Lista(v)),
                        Some(Valor::Inteiro(ini)),
                        Some(Valor::Inteiro(fim)),
                    ) => {
                        let ini = (ini as usize).min(v.len());
                        let fim = (fim as usize).min(v.len());
                        Ok(Valor::Lista(v[ini..fim].to_vec()))
                    }
                    _ => Err("fatiar(lista, inicio, fim)".to_string()),
                }
            }

            // -- Mapas ---------------------------------------------------------
            "mapa" => Ok(Valor::Mapa(HashMap::new())),
            "mapa_definir" => {
                let (m, k, v) = arg3(args, "mapa_definir")?;
                match (m, k) {
                    (Valor::Mapa(mut m), Valor::Texto(k)) => {
                        m.insert(k, v);
                        Ok(Valor::Mapa(m))
                    }
                    _ => Err("mapa_definir(mapa, \"chave\", valor)".to_string()),
                }
            }
            "mapa_obter" => {
                let (m, k) = arg2(args, "mapa_obter")?;
                match (m, k) {
                    (Valor::Mapa(m), Valor::Texto(k)) => {
                        Ok(m.get(&k).cloned().unwrap_or(Valor::Nulo))
                    }
                    _ => Err("mapa_obter(mapa, \"chave\")".to_string()),
                }
            }
            "mapa_tem" => {
                let (m, k) = arg2(args, "mapa_tem")?;
                match (m, k) {
                    (Valor::Mapa(m), Valor::Texto(k)) => Ok(Valor::Booleano(m.contains_key(&k))),
                    _ => Err("mapa_tem(mapa, \"chave\")".to_string()),
                }
            }
            "mapa_remover" => {
                let (m, k) = arg2(args, "mapa_remover")?;
                match (m, k) {
                    (Valor::Mapa(mut m), Valor::Texto(k)) => {
                        m.remove(&k);
                        Ok(Valor::Mapa(m))
                    }
                    _ => Err("mapa_remover(mapa, \"chave\")".to_string()),
                }
            }
            "mapa_chaves" => match arg1(args, "mapa_chaves")? {
                Valor::Mapa(m) => {
                    let mut chaves: Vec<Valor> = m.into_keys().map(Valor::Texto).collect();
                    chaves.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
                    Ok(Valor::Lista(chaves))
                }
                v => Err(format!("'mapa_chaves' requer mapa, recebeu {}", v)),
            },
            "mapa_valores" => match arg1(args, "mapa_valores")? {
                Valor::Mapa(m) => Ok(Valor::Lista(m.into_values().collect())),
                v => Err(format!("'mapa_valores' requer mapa, recebeu {}", v)),
            },
            "mapa_para_lista" => match arg1(args, "mapa_para_lista")? {
                Valor::Mapa(m) => {
                    let mut lista: Vec<Valor> = m
                        .into_iter()
                        .map(|(k, v)| {
                            let mut par = HashMap::new();
                            par.insert("chave".to_string(), Valor::Texto(k));
                            par.insert("valor".to_string(), v);
                            Valor::Mapa(par)
                        })
                        .collect();
                    lista.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
                    Ok(Valor::Lista(lista))
                }
                v => Err(format!("'mapa_para_lista' requer mapa, recebeu {}", v)),
            },

            // -- Texto ---------------------------------------------------------
            "maiusculas" => match arg1(args, "maiusculas")? {
                Valor::Texto(s) => Ok(Valor::Texto(s.to_uppercase())),
                v => Err(format!("'maiusculas' requer texto, recebeu {}", v)),
            },
            "minusculas" => match arg1(args, "minusculas")? {
                Valor::Texto(s) => Ok(Valor::Texto(s.to_lowercase())),
                v => Err(format!("'minusculas' requer texto, recebeu {}", v)),
            },
            "aparar" => match arg1(args, "aparar")? {
                Valor::Texto(s) => Ok(Valor::Texto(s.trim().to_string())),
                v => Err(format!("'aparar' requer texto, recebeu {}", v)),
            },
            "aparar_esquerda" => match arg1(args, "aparar_esquerda")? {
                Valor::Texto(s) => Ok(Valor::Texto(s.trim_start().to_string())),
                v => Err(format!("'aparar_esquerda' requer texto, recebeu {}", v)),
            },
            "aparar_direita" => match arg1(args, "aparar_direita")? {
                Valor::Texto(s) => Ok(Valor::Texto(s.trim_end().to_string())),
                v => Err(format!("'aparar_direita' requer texto, recebeu {}", v)),
            },
            "substituir" => {
                let (t, de, para) = arg3(args, "substituir")?;
                match (t, de, para) {
                    (Valor::Texto(t), Valor::Texto(de), Valor::Texto(para)) => {
                        Ok(Valor::Texto(t.replace(&de as &str, &para as &str)))
                    }
                    _ => Err("substituir(texto, de, para)".to_string()),
                }
            }
            "dividir" => {
                let (t, sep) = arg2(args, "dividir")?;
                match (t, sep) {
                    (Valor::Texto(t), Valor::Texto(sep)) => Ok(Valor::Lista(
                        t.split(&sep as &str)
                            .map(|s| Valor::Texto(s.to_string()))
                            .collect(),
                    )),
                    _ => Err("dividir(texto, separador)".to_string()),
                }
            }
            "juntar" => {
                let (lista, sep) = arg2(args, "juntar")?;
                match (lista, sep) {
                    (Valor::Lista(v), Valor::Texto(sep)) => {
                        let partes: Vec<String> = v.iter().map(|x| x.to_string()).collect();
                        Ok(Valor::Texto(partes.join(&sep)))
                    }
                    _ => Err("juntar(lista, separador)".to_string()),
                }
            }
            "comeca_com" | "começa_com" => {
                let (t, prefixo) = arg2(args, "comeca_com")?;
                match (t, prefixo) {
                    (Valor::Texto(t), Valor::Texto(p)) => {
                        Ok(Valor::Booleano(t.starts_with(&p as &str)))
                    }
                    _ => Err("comeca_com(texto, prefixo)".to_string()),
                }
            }
            "termina_com" => {
                let (t, sufixo) = arg2(args, "termina_com")?;
                match (t, sufixo) {
                    (Valor::Texto(t), Valor::Texto(s)) => {
                        Ok(Valor::Booleano(t.ends_with(&s as &str)))
                    }
                    _ => Err("termina_com(texto, sufixo)".to_string()),
                }
            }
            "contem_texto" => {
                let (t, sub) = arg2(args, "contem_texto")?;
                match (t, sub) {
                    (Valor::Texto(t), Valor::Texto(s)) => {
                        Ok(Valor::Booleano(t.contains(&s as &str)))
                    }
                    _ => Err("contem_texto(texto, substring)".to_string()),
                }
            }
            "posicao" => {
                let (t, sub) = arg2(args, "posicao")?;
                match (t, sub) {
                    (Valor::Texto(t), Valor::Texto(s)) => {
                        let pos = t
                            .find(&s as &str)
                            .map(|i| t[..i].chars().count() as i64)
                            .unwrap_or(-1);
                        Ok(Valor::Inteiro(pos))
                    }
                    _ => Err("posicao(texto, substring)".to_string()),
                }
            }
            "sub_texto" => {
                let mut it = args.into_iter();
                match (it.next(), it.next(), it.next()) {
                    (
                        Some(Valor::Texto(t)),
                        Some(Valor::Inteiro(ini)),
                        Some(Valor::Inteiro(fim)),
                    ) => {
                        let chars: Vec<char> = t.chars().collect();
                        let ini = (ini as usize).min(chars.len());
                        let fim = (fim as usize).min(chars.len());
                        Ok(Valor::Texto(chars[ini..fim].iter().collect()))
                    }
                    (Some(Valor::Texto(t)), Some(Valor::Inteiro(ini)), None) => {
                        let chars: Vec<char> = t.chars().collect();
                        let ini = (ini as usize).min(chars.len());
                        Ok(Valor::Texto(chars[ini..].iter().collect()))
                    }
                    _ => {
                        Err("sub_texto(texto, inicio) ou sub_texto(texto, inicio, fim)".to_string())
                    }
                }
            }
            "inverter_texto" => match arg1(args, "inverter_texto")? {
                Valor::Texto(s) => Ok(Valor::Texto(s.chars().rev().collect())),
                v => Err(format!("'inverter_texto' requer texto, recebeu {}", v)),
            },
            "repetir" => {
                let (t, n) = arg2(args, "repetir")?;
                match (t, n) {
                    (Valor::Texto(s), Valor::Inteiro(n)) => {
                        Ok(Valor::Texto(s.repeat(n.max(0) as usize)))
                    }
                    _ => Err("repetir(texto, n)".to_string()),
                }
            }
            "html_escapar" => match arg1(args, "html_escapar")? {
                Valor::Texto(s) => Ok(Valor::Texto(
                    s.replace('&', "&amp;")
                        .replace('<', "&lt;")
                        .replace('>', "&gt;")
                        .replace('"', "&quot;")
                        .replace('\'', "&#39;"),
                )),
                v => Ok(Valor::Texto(
                    v.to_string()
                        .replace('&', "&amp;")
                        .replace('<', "&lt;")
                        .replace('>', "&gt;"),
                )),
            },
            "url_codificar" => match arg1(args, "url_codificar")? {
                Valor::Texto(s) => {
                    let encoded: String = s
                        .bytes()
                        .flat_map(|b| {
                            if b.is_ascii_alphanumeric()
                                || b == b'-'
                                || b == b'_'
                                || b == b'.'
                                || b == b'~'
                            {
                                vec![b as char]
                            } else {
                                format!("%{:02X}", b).chars().collect()
                            }
                        })
                        .collect();
                    Ok(Valor::Texto(encoded))
                }
                v => Err(format!("'url_codificar' requer texto, recebeu {}", v)),
            },
            "url_decodificar_texto" => match arg1(args, "url_decodificar_texto")? {
                Valor::Texto(s) => Ok(Valor::Texto(url_decodificar_simples(&s))),
                v => Err(format!(
                    "'url_decodificar_texto' requer texto, recebeu {}",
                    v
                )),
            },

            // -- Matematica ----------------------------------------------------
            "raiz" => match arg1(args, "raiz")? {
                Valor::Inteiro(n) => Ok(Valor::Numero((n as f64).sqrt())),
                Valor::Numero(n) => Ok(Valor::Numero(n.sqrt())),
                v => Err(format!("'raiz' requer numero, recebeu {}", v)),
            },
            "potencia" => {
                let (base, exp) = arg2(args, "potencia")?;
                match (base, exp) {
                    (Valor::Inteiro(b), Valor::Inteiro(e)) if e >= 0 => {
                        Ok(Valor::Inteiro(b.pow(e as u32)))
                    }
                    (Valor::Inteiro(b), Valor::Inteiro(e)) => {
                        Ok(Valor::Numero((b as f64).powf(e as f64)))
                    }
                    (Valor::Inteiro(b), Valor::Numero(e)) => Ok(Valor::Numero((b as f64).powf(e))),
                    (Valor::Numero(b), Valor::Inteiro(e)) => Ok(Valor::Numero(b.powf(e as f64))),
                    (Valor::Numero(b), Valor::Numero(e)) => Ok(Valor::Numero(b.powf(e))),
                    _ => Err("potencia(base, expoente)".to_string()),
                }
            }
            "absoluto" => match arg1(args, "absoluto")? {
                Valor::Inteiro(n) => Ok(Valor::Inteiro(n.abs())),
                Valor::Numero(n) => Ok(Valor::Numero(n.abs())),
                v => Err(format!("'absoluto' requer numero, recebeu {}", v)),
            },
            "arredondar" => {
                let mut it = args.into_iter();
                match (it.next(), it.next()) {
                    (Some(Valor::Inteiro(n)), None) => Ok(Valor::Inteiro(n)),
                    (Some(Valor::Inteiro(n)), Some(Valor::Inteiro(casas))) => {
                        let fator = 10f64.powi(casas as i32);
                        Ok(Valor::Numero(((n as f64) * fator).round() / fator))
                    }
                    (Some(Valor::Numero(n)), Some(Valor::Inteiro(casas))) => {
                        let fator = 10f64.powi(casas as i32);
                        Ok(Valor::Numero((n * fator).round() / fator))
                    }
                    (Some(Valor::Numero(n)), None) => Ok(Valor::Numero(n.round())),
                    (Some(Valor::Numero(n)), Some(Valor::Numero(casas))) => {
                        let fator = 10f64.powi(casas as i32);
                        Ok(Valor::Numero((n * fator).round() / fator))
                    }
                    _ => Err("arredondar(n) ou arredondar(n, casas_decimais)".to_string()),
                }
            }
            "piso" => match arg1(args, "piso")? {
                Valor::Inteiro(n) => Ok(Valor::Inteiro(n)),
                Valor::Numero(n) => Ok(Valor::Numero(n.floor())),
                v => Err(format!("'piso' requer numero, recebeu {}", v)),
            },
            "teto" => match arg1(args, "teto")? {
                Valor::Inteiro(n) => Ok(Valor::Inteiro(n)),
                Valor::Numero(n) => Ok(Valor::Numero(n.ceil())),
                v => Err(format!("'teto' requer numero, recebeu {}", v)),
            },
            "minimo" => {
                let (a, b) = arg2(args, "minimo")?;
                match (a, b) {
                    (Valor::Inteiro(x), Valor::Inteiro(y)) => Ok(Valor::Inteiro(x.min(y))),
                    (Valor::Numero(x), Valor::Numero(y)) => Ok(Valor::Numero(x.min(y))),
                    (a, b)
                        if numero_f64(a.clone()).is_some() && numero_f64(b.clone()).is_some() =>
                    {
                        let (x, y) = numeros_f64(a, b, "minimo")?;
                        Ok(Valor::Numero(x.min(y)))
                    }
                    _ => Err("minimo(a, b) requer numeros".to_string()),
                }
            }
            "maximo" => {
                let (a, b) = arg2(args, "maximo")?;
                match (a, b) {
                    (Valor::Inteiro(x), Valor::Inteiro(y)) => Ok(Valor::Inteiro(x.max(y))),
                    (Valor::Numero(x), Valor::Numero(y)) => Ok(Valor::Numero(x.max(y))),
                    (a, b)
                        if numero_f64(a.clone()).is_some() && numero_f64(b.clone()).is_some() =>
                    {
                        let (x, y) = numeros_f64(a, b, "maximo")?;
                        Ok(Valor::Numero(x.max(y)))
                    }
                    _ => Err("maximo(a, b) requer numeros".to_string()),
                }
            }
            "aleatorio" => Ok(Valor::Numero(aleatorio_f64())),
            "aleatorio_inteiro" => {
                let (a, b) = arg2(args, "aleatorio_inteiro")?;
                match (a, b) {
                    (Valor::Inteiro(min), Valor::Inteiro(max)) => {
                        let r = aleatorio_f64();
                        Ok(Valor::Inteiro(
                            (min as f64 + r * (max - min + 1) as f64).floor() as i64,
                        ))
                    }
                    (Valor::Numero(min), Valor::Numero(max)) => {
                        let r = aleatorio_f64();
                        let resultado = (min + r * (max - min + 1.0)).floor().min(max);
                        Ok(Valor::Numero(resultado))
                    }
                    _ => Err("aleatorio_inteiro(min, max)".to_string()),
                }
            }
            "seno" => {
                let v = arg1(args, "seno")?;
                Ok(Valor::Numero(arg_numero(v, "seno")?.sin()))
            }
            "cosseno" => {
                let v = arg1(args, "cosseno")?;
                Ok(Valor::Numero(arg_numero(v, "cosseno")?.cos()))
            }
            "tangente" => {
                let v = arg1(args, "tangente")?;
                Ok(Valor::Numero(arg_numero(v, "tangente")?.tan()))
            }
            "logaritmo" => match arg1(args, "logaritmo")? {
                Valor::Inteiro(n) => Ok(Valor::Numero((n as f64).ln())),
                Valor::Numero(n) => Ok(Valor::Numero(n.ln())),
                v => Err(format!("'logaritmo' requer numero, recebeu {}", v)),
            },
            "pi" => Ok(Valor::Numero(std::f64::consts::PI)),
            "infinito" => Ok(Valor::Numero(f64::INFINITY)),
            "eh_numero" => match arg1(args, "eh_numero")? {
                Valor::Inteiro(_) | Valor::Numero(_) => Ok(Valor::Booleano(true)),
                Valor::Texto(s) => Ok(Valor::Booleano(s.trim().parse::<f64>().is_ok())),
                _ => Ok(Valor::Booleano(false)),
            },

            // -- Data e hora ---------------------------------------------------
            "formatar_numero" => {
                let (valor, casas) = arg2(args, "formatar_numero")?;
                let numero = arg_numero(valor, "formatar_numero")?;
                let casas = match casas {
                    Valor::Inteiro(n) if n >= 0 => n as usize,
                    v => {
                        return Err(format!(
                            "'formatar_numero' requer casas inteiras, recebeu {}",
                            v
                        ))
                    }
                };
                Ok(Valor::Texto(formatar_numero_br(numero, casas)))
            }
            "formatar_data" => {
                let (data, formato) = arg2(args, "formatar_data")?;
                let data = match data {
                    Valor::Texto(s) => s,
                    v => return Err(format!("data invalida: {}", v)),
                };
                let formato = match formato {
                    Valor::Texto(s) => s,
                    v => return Err(format!("formato invalido: {}", v)),
                };
                Ok(Valor::Texto(formatar_data_simples(&data, &formato)?))
            }
            "timestamp" => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let t = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                Ok(Valor::Numero(t.as_secs_f64()))
            }
            "data_hora" => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let secs = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let seg_dia = secs % 86400;
                let h = seg_dia / 3600;
                let m = (seg_dia % 3600) / 60;
                let s = seg_dia % 60;
                let (ano, mes, dia) = timestamp_para_data(secs);
                Ok(Valor::Texto(format!(
                    "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
                    ano, mes, dia, h, m, s
                )))
            }
            "dormir" => match arg1(args, "dormir")? {
                Valor::Inteiro(ms) => {
                    std::thread::sleep(std::time::Duration::from_millis(ms.max(0) as u64));
                    Ok(Valor::Nulo)
                }
                Valor::Numero(ms) => {
                    std::thread::sleep(std::time::Duration::from_millis(ms as u64));
                    Ok(Valor::Nulo)
                }
                v => Err(format!("'dormir' requer numero (ms), recebeu {}", v)),
            },

            // -- Arquivos ------------------------------------------------------
            "ler_arquivo" => match arg1(args, "ler_arquivo")? {
                Valor::Texto(caminho) => std::fs::read_to_string(&caminho)
                    .map(Valor::Texto)
                    .map_err(|e| format!("ler_arquivo('{}'): {}", caminho, e)),
                v => Err(format!("'ler_arquivo' requer texto, recebeu {}", v)),
            },
            "escrever_arquivo" => {
                let (caminho, conteudo) = arg2(args, "escrever_arquivo")?;
                match (caminho, conteudo) {
                    (Valor::Texto(p), conteudo) => {
                        std::fs::write(&p, conteudo.to_string().as_bytes())
                            .map(|_| Valor::Booleano(true))
                            .map_err(|e| format!("escrever_arquivo('{}'): {}", p, e))
                    }
                    _ => Err("escrever_arquivo(caminho, conteudo)".to_string()),
                }
            }
            "escrever_arquivo_bytes" => {
                let (caminho, conteudo) = arg2(args, "escrever_arquivo_bytes")?;
                match (caminho, conteudo) {
                    (Valor::Texto(p), Valor::Bytes(b)) => std::fs::write(&p, b.as_ref())
                        .map(|_| Valor::Booleano(true))
                        .map_err(|e| format!("escrever_arquivo_bytes('{}'): {}", p, e)),
                    _ => Err("escrever_arquivo_bytes(caminho, bytes)".to_string()),
                }
            }
            "acrescentar_arquivo" => {
                let (caminho, conteudo) = arg2(args, "acrescentar_arquivo")?;
                match (caminho, conteudo) {
                    (Valor::Texto(p), conteudo) => {
                        use std::io::Write as _;
                        let mut f = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(&p)
                            .map_err(|e| format!("acrescentar_arquivo('{}'): {}", p, e))?;
                        write!(f, "{}", conteudo).map_err(|e| e.to_string())?;
                        Ok(Valor::Booleano(true))
                    }
                    _ => Err("acrescentar_arquivo(caminho, conteudo)".to_string()),
                }
            }
            "arquivo_existe" => match arg1(args, "arquivo_existe")? {
                Valor::Texto(p) => Ok(Valor::Booleano(std::path::Path::new(&p).exists())),
                v => Err(format!("'arquivo_existe' requer texto, recebeu {}", v)),
            },
            "apagar_arquivo" => match arg1(args, "apagar_arquivo")? {
                Valor::Texto(p) => std::fs::remove_file(&p)
                    .map(|_| Valor::Booleano(true))
                    .map_err(|e| format!("apagar_arquivo('{}'): {}", p, e)),
                v => Err(format!("'apagar_arquivo' requer texto, recebeu {}", v)),
            },
            "listar_arquivos" => match arg1(args, "listar_arquivos")? {
                Valor::Texto(p) => {
                    let entradas = std::fs::read_dir(&p)
                        .map_err(|e| format!("listar_arquivos('{}'): {}", p, e))?;
                    let mut lista = Vec::new();
                    for entrada in entradas.flatten() {
                        if let Ok(nome) = entrada.file_name().into_string() {
                            lista.push(Valor::Texto(nome));
                        }
                    }
                    lista.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
                    Ok(Valor::Lista(lista))
                }
                v => Err(format!("'listar_arquivos' requer texto, recebeu {}", v)),
            },
            "criar_diretorio" => match arg1(args, "criar_diretorio")? {
                Valor::Texto(p) => std::fs::create_dir_all(&p)
                    .map(|_| Valor::Booleano(true))
                    .map_err(|e| format!("criar_diretorio('{}'): {}", p, e)),
                v => Err(format!("'criar_diretorio' requer texto, recebeu {}", v)),
            },
            "eh_arquivo" => match arg1(args, "eh_arquivo")? {
                Valor::Texto(p) => Ok(Valor::Booleano(std::path::Path::new(&p).is_file())),
                v => Err(format!("'eh_arquivo' requer texto, recebeu {}", v)),
            },
            "eh_diretorio" => match arg1(args, "eh_diretorio")? {
                Valor::Texto(p) => Ok(Valor::Booleano(std::path::Path::new(&p).is_dir())),
                v => Err(format!("'eh_diretorio' requer texto, recebeu {}", v)),
            },

            // -- CSV -----------------------------------------------------------
            "csv_parsear" => match arg1(args, "csv_parsear")? {
                Valor::Texto(s) => Ok(Valor::Lista(csv_parsear_texto(&s))),
                v => Err(format!("'csv_parsear' requer texto, recebeu {}", v)),
            },
            "csv_ler" => match arg1(args, "csv_ler")? {
                Valor::Texto(p) => {
                    let s = std::fs::read_to_string(&p)
                        .map_err(|e| format!("csv_ler('{}'): {}", p, e))?;
                    Ok(Valor::Lista(csv_parsear_texto(&s)))
                }
                v => Err(format!("'csv_ler' requer texto, recebeu {}", v)),
            },
            "csv_ler_mapa" => match arg1(args, "csv_ler_mapa")? {
                Valor::Texto(p) => {
                    let s = std::fs::read_to_string(&p)
                        .map_err(|e| format!("csv_ler_mapa('{}'): {}", p, e))?;
                    let linhas = csv_parsear_texto(&s);
                    if linhas.is_empty() {
                        return Ok(Valor::Lista(vec![]));
                    }
                    let cabecalho: Vec<String> = match &linhas[0] {
                        Valor::Lista(cols) => cols.iter().map(|c| c.to_string()).collect(),
                        _ => return Err("csv_ler_mapa: linha de cabecalho invalida".to_string()),
                    };
                    let resultado: Vec<Valor> = linhas[1..]
                        .iter()
                        .map(|linha| {
                            let mut map = std::collections::HashMap::new();
                            if let Valor::Lista(cols) = linha {
                                for (i, chave) in cabecalho.iter().enumerate() {
                                    let val =
                                        cols.get(i).cloned().unwrap_or(Valor::Texto(String::new()));
                                    map.insert(chave.clone(), val);
                                }
                            }
                            Valor::Mapa(map)
                        })
                        .collect();
                    Ok(Valor::Lista(resultado))
                }
                v => Err(format!("'csv_ler_mapa' requer texto, recebeu {}", v)),
            },
            "csv_serializar" => match arg1(args, "csv_serializar")? {
                Valor::Lista(linhas) => Ok(Valor::Texto(csv_serializar_linhas(&linhas))),
                v => Err(format!("'csv_serializar' requer lista, recebeu {}", v)),
            },
            "csv_escrever" => {
                let (caminho, dados) = arg2(args, "csv_escrever")?;
                match (caminho, dados) {
                    (Valor::Texto(p), Valor::Lista(linhas)) => {
                        let conteudo = csv_serializar_linhas(&linhas);
                        std::fs::write(&p, conteudo.as_bytes())
                            .map(|_| Valor::Booleano(true))
                            .map_err(|e| format!("csv_escrever('{}'): {}", p, e))
                    }
                    _ => Err("csv_escrever(caminho, lista_de_listas)".to_string()),
                }
            }

            // -- JSON ----------------------------------------------------------
            "json_serializar" => Ok(Valor::Texto(valor_para_json(&arg1(
                args,
                "json_serializar",
            )?))),
            "json_deserializar" => match arg1(args, "json_deserializar")? {
                Valor::Texto(s) => json_para_valor(s.trim()),
                v => Err(format!("'json_deserializar' requer texto, recebeu {}", v)),
            },

            // -- Sessao / Cookie (web) — repositorio in-memory thread-safe ------
            "sessao_iniciar" => {
                let id = crate::sessoes::obter_sessao_atual();
                let id = if id.is_empty() { novo_id_sessao() } else { id };
                // Garante que a sessao existe no repositorio
                crate::sessoes::repo().write().unwrap().garantir(&id);
                Ok(Valor::Texto(id))
            }
            "sessao_obter" => {
                let chave = match arg1(args, "sessao_obter")? {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'sessao_obter' requer texto, recebeu {}", v)),
                };
                let id = crate::sessoes::obter_sessao_atual();
                if id.is_empty() {
                    return Ok(Valor::Nulo);
                }
                Ok(crate::sessoes::repo()
                    .read()
                    .unwrap()
                    .obter(&id, &chave)
                    .unwrap_or(Valor::Nulo))
            }
            "sessao_definir" => {
                let (chave, valor) = arg2(args, "sessao_definir")?;
                let chave = match chave {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'sessao_definir' chave deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let id = crate::sessoes::obter_sessao_atual();
                let id = if id.is_empty() { novo_id_sessao() } else { id };
                crate::sessoes::repo()
                    .write()
                    .unwrap()
                    .definir(&id, chave, valor);
                Ok(Valor::Nulo)
            }
            "sessao_remover" => {
                let chave = match arg1(args, "sessao_remover")? {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'sessao_remover' requer texto, recebeu {}", v)),
                };
                let id = crate::sessoes::obter_sessao_atual();
                if id.is_empty() {
                    return Ok(Valor::Nulo);
                }
                Ok(crate::sessoes::repo()
                    .write()
                    .unwrap()
                    .remover(&id, &chave)
                    .unwrap_or(Valor::Nulo))
            }
            "sessao_destruir" => {
                let id = crate::sessoes::obter_sessao_atual();
                if !id.is_empty() {
                    crate::sessoes::repo().write().unwrap().destruir(&id);
                }
                Ok(Valor::Nulo)
            }
            "sessao_renovar" => {
                let id = crate::sessoes::obter_sessao_atual();
                if !id.is_empty() {
                    let minutos: Option<u64> = if args.is_empty() {
                        None
                    } else {
                        match arg1(args, "sessao_renovar")? {
                            Valor::Inteiro(n) if n > 0 => Some(n as u64),
                            Valor::Numero(n) if n > 0.0 => Some(n as u64),
                            _ => None,
                        }
                    };
                    let repo_arc = crate::sessoes::repo();
                    let mut repo = repo_arc.write().unwrap();
                    if let Some(min) = minutos {
                        repo.definir_ttl(&id, min);
                    } else {
                        repo.renovar(&id);
                    }
                }
                Ok(Valor::Nulo)
            }
            "sessao_regenerar" => {
                let antigo = crate::sessoes::obter_sessao_atual();
                let novo = novo_id_sessao();
                crate::sessoes::repo()
                    .write()
                    .unwrap()
                    .migrar(&antigo, &novo);
                crate::sessoes::definir_sessao_atual(novo.clone());
                let secure = std::env::var("PEP_COOKIE_SECURE")
                    .ok()
                    .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
                let cookie = format!(
                    "pep_sessao={}; Path=/; HttpOnly; SameSite=Lax{}",
                    novo,
                    if secure { "; Secure" } else { "" }
                );
                HDRS_HTTP.with(|h| h.borrow_mut().push(("Set-Cookie".to_string(), cookie)));
                Ok(Valor::Texto(novo))
            }
            "sessao_listar_chaves" => {
                let id = crate::sessoes::obter_sessao_atual();
                if id.is_empty() {
                    return Ok(Valor::Lista(vec![]));
                }
                let chaves = crate::sessoes::repo().read().unwrap().listar_chaves(&id);
                Ok(Valor::Lista(chaves.into_iter().map(Valor::Texto).collect()))
            }
            "sessao_obter_tudo" => {
                let id = crate::sessoes::obter_sessao_atual();
                if id.is_empty() {
                    return Ok(Valor::Mapa(HashMap::new()));
                }
                Ok(Valor::Mapa(
                    crate::sessoes::repo().read().unwrap().obter_todos(&id),
                ))
            }
            "csrf_token" => {
                let id = crate::sessoes::obter_sessao_atual();
                let id = if id.is_empty() { novo_id_sessao() } else { id };
                let token = bytes_aleatorios_hex(32).unwrap_or_else(|_| {
                    format!("{:016x}{:016x}", aleatorio_u64(), aleatorio_u64())
                });
                crate::sessoes::repo().write().unwrap().definir(
                    &id,
                    "__csrf__".to_string(),
                    Valor::Texto(token.clone()),
                );
                Ok(Valor::Texto(token))
            }
            "csrf_verificar" => {
                let token = match arg1(args, "csrf_verificar")? {
                    Valor::Texto(s) => s,
                    _ => return Ok(Valor::Booleano(false)),
                };
                let id = crate::sessoes::obter_sessao_atual();
                if id.is_empty() {
                    return Ok(Valor::Booleano(false));
                }
                let esperado = crate::sessoes::repo()
                    .read()
                    .unwrap()
                    .obter(&id, "__csrf__");
                let ok = matches!(&esperado, Some(Valor::Texto(t)) if *t == token);
                if ok {
                    // Token de uso unico: remove apos verificacao
                    crate::sessoes::repo()
                        .write()
                        .unwrap()
                        .remover(&id, "__csrf__");
                }
                Ok(Valor::Booleano(ok))
            }
            "cookie_obter" => {
                let nome = match arg1(args, "cookie_obter")? {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'cookie_obter' requer texto, recebeu {}", v)),
                };
                if let Some(Valor::Mapa(cookies)) = env.obter("_COOKIE") {
                    return Ok(cookies.get(&nome).cloned().unwrap_or(Valor::Nulo));
                }
                let cookie_str = std::env::var("PEP_COOKIE").unwrap_or_default();
                for par in cookie_str.split(';') {
                    let par = par.trim();
                    if let Some((k, v)) = par.split_once('=') {
                        if k.trim() == nome {
                            return Ok(Valor::Texto(url_decodificar_simples(v.trim())));
                        }
                    }
                }
                Ok(Valor::Nulo)
            }
            "cookie_definir" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err("cookie_definir(nome, valor, opcoes?)".to_string());
                }
                let nome = match &args[0] {
                    Valor::Texto(s) => s.clone(),
                    v => {
                        return Err(format!(
                            "cookie_definir: nome deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let valor = match &args[1] {
                    Valor::Texto(s) => s.clone(),
                    v => v.to_string(),
                };
                if nome.is_empty()
                    || nome
                        .chars()
                        .any(|c| c.is_control() || matches!(c, ' ' | ';' | ',' | '='))
                    || valor.contains(['\r', '\n', ';'])
                {
                    return Err("cookie_definir: nome ou valor invalido".to_string());
                }
                let mut path = "/".to_string();
                let mut max_age: Option<i64> = None;
                let mut http_only = true;
                let mut secure = false;
                let mut same_site = "Lax".to_string();
                if let Some(Valor::Mapa(opcoes)) = args.get(2) {
                    if let Some(Valor::Texto(v)) = opcoes.get("caminho") {
                        path = v.clone();
                    }
                    if let Some(v) = opcoes.get("max_idade") {
                        max_age = match v {
                            Valor::Inteiro(n) => Some(*n),
                            Valor::Numero(n) => Some(*n as i64),
                            _ => None,
                        };
                    }
                    if let Some(Valor::Booleano(v)) = opcoes.get("http_only") {
                        http_only = *v;
                    }
                    if let Some(Valor::Booleano(v)) = opcoes.get("seguro") {
                        secure = *v;
                    }
                    if let Some(Valor::Texto(v)) = opcoes.get("same_site") {
                        same_site = v.clone();
                    }
                }
                if path.contains(['\r', '\n', ';'])
                    || !matches!(same_site.as_str(), "Lax" | "Strict" | "None")
                {
                    return Err("cookie_definir: opcoes invalidas".to_string());
                }
                let mut cookie =
                    format!("{}={}; Path={}; SameSite={}", nome, valor, path, same_site);
                if let Some(v) = max_age {
                    cookie.push_str(&format!("; Max-Age={}", v));
                }
                if http_only {
                    cookie.push_str("; HttpOnly");
                }
                if secure {
                    cookie.push_str("; Secure");
                }
                HDRS_HTTP.with(|h| h.borrow_mut().push(("Set-Cookie".to_string(), cookie)));
                Ok(Valor::Nulo)
            }

            "rota" => {
                if args.len() < 3 {
                    return Err("'rota' requer 3 argumentos: metodo, caminho, handler".to_string());
                }
                let metodo = match args[0].clone() {
                    Valor::Texto(s) => s.to_uppercase(),
                    v => return Err(format!("'rota' metodo deve ser texto, recebeu {}", v)),
                };
                let padrao = match args[1].clone() {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'rota' caminho deve ser texto, recebeu {}", v)),
                };
                let vm_handler = match args[2].clone() {
                    Valor::Funcao {
                        parametros, corpo, ..
                    } => {
                        let params: Vec<std::sync::Arc<str>> = parametros
                            .iter()
                            .map(|p| std::sync::Arc::from(p.nome.as_str()))
                            .collect();
                        let ops = crate::compilador::compilar_corpo_funcao(&corpo)?;
                        crate::vm::VmValor::Funcao { params, corpo: ops }
                    }
                    v => return Err(format!("'rota' handler deve ser funcao, recebeu {}", v)),
                };
                crate::servidor::registrar_rota(metodo, padrao, vm_handler);
                Ok(Valor::Nulo)
            }

            "cabecalho" => {
                let (nome_v, valor) = arg2(args, "cabecalho")?;
                let nome_h = match nome_v {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'cabecalho' requer nome em texto, recebeu {}", v)),
                };
                let valor_h = valor.to_string();
                if nome_h.is_empty()
                    || nome_h.contains(['\r', '\n', ':'])
                    || valor_h.contains(['\r', '\n'])
                {
                    return Err("cabecalho: nome ou valor invalido".to_string());
                }
                if MODO_SERVIDOR.with(|m| m.get()) {
                    HDRS_HTTP.with(|h| h.borrow_mut().push((nome_h, valor_h)));
                } else {
                    println!("\u{1e}PEP_HEADER:{}: {}", nome_h, valor);
                }
                Ok(Valor::Nulo)
            }
            "status" => {
                let codigo = match arg1(args, "status")? {
                    Valor::Inteiro(n) => n,
                    Valor::Numero(n) => n as i64,
                    v => return Err(format!("'status' requer numero, recebeu {}", v)),
                };
                if !(100..=599).contains(&codigo) {
                    return Err("status: codigo deve estar entre 100 e 599".to_string());
                }
                if MODO_SERVIDOR.with(|m| m.get()) {
                    STATUS_HTTP.with(|s| s.set(codigo as u16));
                } else {
                    println!("\u{1e}PEP_STATUS:{}", codigo);
                }
                Ok(Valor::Nulo)
            }
            "redirecionar" => {
                let url = match arg1(args, "redirecionar")? {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'redirecionar' requer texto, recebeu {}", v)),
                };
                if url.contains(['\r', '\n']) {
                    return Err("redirecionar: URL invalida".to_string());
                }
                if MODO_SERVIDOR.with(|m| m.get()) {
                    STATUS_HTTP.with(|s| s.set(302));
                    HDRS_HTTP.with(|h| h.borrow_mut().push(("Location".to_string(), url)));
                } else {
                    println!("\u{1e}PEP_STATUS:302");
                    println!("\u{1e}PEP_HEADER:Location: {}", url);
                }
                Ok(Valor::Nulo)
            }
            "json_responder" => {
                let valor = arg1(args, "json_responder")?;
                let json = valor_para_json(&valor);
                if MODO_SERVIDOR.with(|m| m.get()) {
                    HDRS_HTTP.with(|h| {
                        h.borrow_mut().push((
                            "Content-Type".to_string(),
                            "application/json; charset=utf-8".to_string(),
                        ))
                    });
                    SAIDA_HTTP.with(|s| s.borrow_mut().extend_from_slice(json.as_bytes()));
                } else {
                    println!("\u{1e}PEP_HEADER:Content-Type: application/json; charset=utf-8");
                    print!("{}", json);
                }
                Ok(Valor::Nulo)
            }
            // -- Vetores (sobre listas de numeros) ----------------------------
            "vec_soma" | "vec_sub" | "vec_mul" | "vec_div" => {
                let (a, b) = arg2(args, nome)?;
                let va = lista_f64(&a, nome)?;
                // b pode ser lista (elemento-a-elemento) ou escalar
                let op: Box<dyn Fn(f64, f64) -> f64> = match nome {
                    "vec_soma" => Box::new(|x, y| x + y),
                    "vec_sub" => Box::new(|x, y| x - y),
                    "vec_mul" => Box::new(|x, y| x * y),
                    _ => Box::new(|x, y| x / y),
                };
                let resultado: Result<Vec<f64>, String> = match &b {
                    Valor::Lista(_) => {
                        let vb = lista_f64(&b, nome)?;
                        if va.len() != vb.len() {
                            return Err(format!(
                                "{}: vetores de tamanhos diferentes ({} vs {})",
                                nome,
                                va.len(),
                                vb.len()
                            ));
                        }
                        Ok(va.iter().zip(vb.iter()).map(|(x, y)| op(*x, *y)).collect())
                    }
                    _ => {
                        let esc = to_f64(&b, nome)?;
                        Ok(va.iter().map(|x| op(*x, esc)).collect())
                    }
                };
                Ok(Valor::Lista(
                    resultado?.into_iter().map(Valor::Numero).collect(),
                ))
            }
            "produto_interno" => {
                let (a, b) = arg2(args, "produto_interno")?;
                let va = lista_f64(&a, "produto_interno")?;
                let vb = lista_f64(&b, "produto_interno")?;
                if va.len() != vb.len() {
                    return Err(format!("produto_interno: vetores de tamanhos diferentes"));
                }
                Ok(Valor::Numero(
                    va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum(),
                ))
            }
            "norma" => {
                let v = lista_f64(&arg1(args, "norma")?, "norma")?;
                Ok(Valor::Numero(v.iter().map(|x| x * x).sum::<f64>().sqrt()))
            }
            "normalizar" => {
                let v = lista_f64(&arg1(args, "normalizar")?, "normalizar")?;
                let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if n == 0.0 {
                    return Err("normalizar: vetor nulo nao pode ser normalizado".to_string());
                }
                Ok(Valor::Lista(
                    v.into_iter().map(|x| Valor::Numero(x / n)).collect(),
                ))
            }
            "produto_vetorial" => {
                let (a, b) = arg2(args, "produto_vetorial")?;
                let va = lista_f64(&a, "produto_vetorial")?;
                let vb = lista_f64(&b, "produto_vetorial")?;
                if va.len() != 3 || vb.len() != 3 {
                    return Err("produto_vetorial: requer dois vetores 3D".to_string());
                }
                Ok(Valor::Lista(vec![
                    Valor::Numero(va[1] * vb[2] - va[2] * vb[1]),
                    Valor::Numero(va[2] * vb[0] - va[0] * vb[2]),
                    Valor::Numero(va[0] * vb[1] - va[1] * vb[0]),
                ]))
            }

            // -- Matrizes / Tensores (API legada + nova sobre Tensor) ---------
            "matriz" => {
                let n = args.len();
                if n < 2 {
                    return Err("'matriz' requer linhas, colunas".to_string());
                }
                let linhas = to_usize(&args[0], "matriz")?;
                let colunas = to_usize(&args[1], "matriz")?;
                let init = if n > 2 {
                    to_f64(&args[2], "matriz")?
                } else {
                    0.0
                };
                Ok(Valor::Tensor {
                    shape: vec![linhas, colunas],
                    dados: Arc::new(vec![init; linhas * colunas]),
                })
            }
            "mat_de" => match arg1(args, "mat_de")? {
                Valor::Lista(linhas_v) => {
                    let nlin = linhas_v.len();
                    if nlin == 0 {
                        return Ok(Valor::Tensor {
                            shape: vec![0, 0],
                            dados: Arc::new(vec![]),
                        });
                    }
                    let ncol = match &linhas_v[0] {
                        Valor::Lista(l) => l.len(),
                        _ => return Err("'mat_de' requer lista de listas".to_string()),
                    };
                    let mut d = Vec::with_capacity(nlin * ncol);
                    for row in &linhas_v {
                        match row {
                            Valor::Lista(l) => {
                                if l.len() != ncol {
                                    return Err(
                                        "'mat_de': linhas com tamanhos diferentes".to_string()
                                    );
                                }
                                for v in l {
                                    d.push(to_f64(v, "mat_de")?);
                                }
                            }
                            _ => return Err("'mat_de' requer lista de listas".to_string()),
                        }
                    }
                    Ok(Valor::Tensor {
                        shape: vec![nlin, ncol],
                        dados: Arc::new(d),
                    })
                }
                v => Err(format!("'mat_de' requer lista, recebeu {}", v)),
            },
            "mat_identidade" => {
                let n = to_usize(&arg1(args, "mat_identidade")?, "mat_identidade")?;
                let mut d = vec![0.0f64; n * n];
                for i in 0..n {
                    d[i * n + i] = 1.0;
                }
                Ok(Valor::Tensor {
                    shape: vec![n, n],
                    dados: Arc::new(d),
                })
            }
            "mat_linhas" => match arg1(args, "mat_linhas")? {
                Valor::Tensor { shape, .. } if !shape.is_empty() => {
                    Ok(Valor::Inteiro(shape[0] as i64))
                }
                v => Err(format!("'mat_linhas' requer tensor, recebeu {}", v)),
            },
            "mat_colunas" => match arg1(args, "mat_colunas")? {
                Valor::Tensor { shape, .. } if shape.len() >= 2 => {
                    Ok(Valor::Inteiro(shape[1] as i64))
                }
                v => Err(format!("'mat_colunas' requer tensor 2D, recebeu {}", v)),
            },
            "mat_obter" => {
                if args.len() < 3 {
                    return Err("'mat_obter' requer tensor, linha, coluna".to_string());
                }
                match &args[0] {
                    Valor::Tensor { shape, dados } if shape.len() == 2 => {
                        let l = to_usize(&args[1], "mat_obter")?;
                        let c = to_usize(&args[2], "mat_obter")?;
                        if l >= shape[0] || c >= shape[1] {
                            return Err(format!(
                                "mat_obter: indice ({},{}) fora de ({}x{})",
                                l, c, shape[0], shape[1]
                            ));
                        }
                        Ok(Valor::Numero(dados[l * shape[1] + c]))
                    }
                    v => Err(format!("'mat_obter' requer tensor 2D, recebeu {}", v)),
                }
            }
            "mat_definir" => {
                if args.len() < 4 {
                    return Err("'mat_definir' requer tensor, linha, coluna, valor".to_string());
                }
                match args[0].clone() {
                    Valor::Tensor { shape, dados } if shape.len() == 2 => {
                        let l = to_usize(&args[1], "mat_definir")?;
                        let c = to_usize(&args[2], "mat_definir")?;
                        let v = to_f64(&args[3], "mat_definir")?;
                        if l >= shape[0] || c >= shape[1] {
                            return Err(format!(
                                "mat_definir: indice ({},{}) fora de ({}x{})",
                                l, c, shape[0], shape[1]
                            ));
                        }
                        let mut d = dados.to_vec();
                        d[l * shape[1] + c] = v;
                        Ok(Valor::Tensor {
                            shape,
                            dados: Arc::new(d),
                        })
                    }
                    v => Err(format!("'mat_definir' requer tensor 2D, recebeu {}", v)),
                }
            }
            "mat_transpor" => match arg1(args, "mat_transpor")? {
                Valor::Tensor { shape, dados } if shape.len() == 2 => {
                    Ok(tensor_transpor_2d(shape[0], shape[1], &dados))
                }
                v => Err(format!("'mat_transpor' requer tensor 2D, recebeu {}", v)),
            },
            "mat_soma" | "mat_sub" => {
                let (a, b) = arg2(args, nome)?;
                match (a, b) {
                    (
                        Valor::Tensor {
                            shape: sa,
                            dados: da,
                        },
                        Valor::Tensor {
                            shape: sb,
                            dados: db,
                        },
                    ) => {
                        if sa != sb {
                            return Err(format!(
                                "{}: shapes incompativeis {:?} vs {:?}",
                                nome, sa, sb
                            ));
                        }
                        let nd: Vec<f64> = if nome == "mat_soma" {
                            da.iter().zip(db.iter()).map(|(x, y)| x + y).collect()
                        } else {
                            da.iter().zip(db.iter()).map(|(x, y)| x - y).collect()
                        };
                        Ok(Valor::Tensor {
                            shape: sa,
                            dados: Arc::new(nd),
                        })
                    }
                    (va, _) => Err(format!("'{}' requer dois tensores, recebeu {}", nome, va)),
                }
            }
            "mat_mul" | "tensor_matmul" => {
                let (a, b) = arg2(args, nome)?;
                match (a, b) {
                    (
                        Valor::Tensor {
                            shape: sa,
                            dados: da,
                        },
                        Valor::Tensor {
                            shape: sb,
                            dados: db,
                        },
                    ) => {
                        if sa.len() != 2 || sb.len() != 2 {
                            return Err(format!(
                                "{}: requer tensores 2D ({}D x {}D)",
                                nome,
                                sa.len(),
                                sb.len()
                            ));
                        }
                        if sa[1] != sb[0] {
                            return Err(format!(
                                "{}: colunas A ({}) != linhas B ({})",
                                nome, sa[1], sb[0]
                            ));
                        }
                        Ok(tensor_matmul_ndarray(sa[0], sa[1], &da, sb[0], sb[1], &db))
                    }
                    (Valor::Tensor { shape, dados }, escalar) => {
                        let esc = to_f64(&escalar, nome)?;
                        Ok(Valor::Tensor {
                            shape,
                            dados: Arc::new(dados.iter().map(|x| x * esc).collect()),
                        })
                    }
                    (va, _) => Err(format!("'{}' requer tensor, recebeu {}", nome, va)),
                }
            }
            "mat_para_lista" | "tensor_para_lista" => match arg1(args, nome)? {
                Valor::Tensor { shape, dados } => Ok(tensor_para_lista_val(&shape, &dados)),
                v => Err(format!("'{}' requer tensor, recebeu {}", nome, v)),
            },

            // -- Tensor n-dimensional -----------------------------------------
            "tensor" | "tensor_zeros" => {
                let shape = shape_de_args(&args, nome)?;
                let total: usize = if shape.is_empty() {
                    0
                } else {
                    shape.iter().product()
                };
                Ok(Valor::Tensor {
                    shape,
                    dados: Arc::new(vec![0.0; total]),
                })
            }
            "tensor_uns" => {
                let shape = shape_de_args(&args, "tensor_uns")?;
                let total: usize = if shape.is_empty() {
                    0
                } else {
                    shape.iter().product()
                };
                Ok(Valor::Tensor {
                    shape,
                    dados: Arc::new(vec![1.0; total]),
                })
            }
            "tensor_de" => {
                if args.len() < 2 {
                    return Err("'tensor_de' requer lista_plana, shape".to_string());
                }
                let d = lista_f64(&args[0], "tensor_de")?;
                let shape = match &args[1] {
                    Valor::Lista(l) => l
                        .iter()
                        .map(|v| to_usize(v, "tensor_de"))
                        .collect::<Result<Vec<_>, _>>()?,
                    _ => {
                        return Err(
                            "'tensor_de': segundo arg deve ser lista de dimensoes".to_string()
                        )
                    }
                };
                let total: usize = if shape.is_empty() {
                    0
                } else {
                    shape.iter().product()
                };
                if d.len() != total {
                    return Err(format!(
                        "tensor_de: {} elementos mas shape requer {}",
                        d.len(),
                        total
                    ));
                }
                Ok(Valor::Tensor {
                    shape,
                    dados: Arc::new(d),
                })
            }
            "tensor_shape" => match arg1(args, "tensor_shape")? {
                Valor::Tensor { shape, .. } => Ok(Valor::Lista(
                    shape.iter().map(|&n| Valor::Inteiro(n as i64)).collect(),
                )),
                v => Err(format!("'tensor_shape' requer tensor, recebeu {}", v)),
            },
            "tensor_ndim" => match arg1(args, "tensor_ndim")? {
                Valor::Tensor { shape, .. } => Ok(Valor::Inteiro(shape.len() as i64)),
                v => Err(format!("'tensor_ndim' requer tensor, recebeu {}", v)),
            },
            "tensor_tamanho" => match arg1(args, "tensor_tamanho")? {
                Valor::Tensor { dados, .. } => Ok(Valor::Inteiro(dados.len() as i64)),
                v => Err(format!("'tensor_tamanho' requer tensor, recebeu {}", v)),
            },
            "tensor_reshape" => {
                if args.len() < 2 {
                    return Err("'tensor_reshape' requer tensor, nova_shape".to_string());
                }
                match args[0].clone() {
                    Valor::Tensor { dados, .. } => {
                        let shape = match &args[1] {
                            Valor::Lista(l) => l
                                .iter()
                                .map(|v| to_usize(v, "tensor_reshape"))
                                .collect::<Result<Vec<_>, _>>()?,
                            _ => {
                                return Err(
                                    "'tensor_reshape': segundo arg deve ser lista de dimensoes"
                                        .to_string(),
                                )
                            }
                        };
                        let total: usize = if shape.is_empty() {
                            0
                        } else {
                            shape.iter().product()
                        };
                        if dados.len() != total {
                            return Err(format!(
                                "tensor_reshape: {} elementos mas nova shape requer {}",
                                dados.len(),
                                total
                            ));
                        }
                        Ok(Valor::Tensor { shape, dados })
                    }
                    v => Err(format!("'tensor_reshape' requer tensor, recebeu {}", v)),
                }
            }
            "tensor_transpor" => match arg1(args, "tensor_transpor")? {
                Valor::Tensor { shape, dados } if shape.len() == 2 => {
                    Ok(tensor_transpor_2d(shape[0], shape[1], &dados))
                }
                Valor::Tensor { shape, .. } => Err(format!(
                    "'tensor_transpor': apenas 2D suportado (recebeu {}D)",
                    shape.len()
                )),
                v => Err(format!("'tensor_transpor' requer tensor, recebeu {}", v)),
            },
            "tensor_soma" | "tensor_sub" | "tensor_mul" | "tensor_div" => {
                let (a, b) = arg2(args, nome)?;
                let op: Box<dyn Fn(f64, f64) -> f64> = match nome {
                    "tensor_soma" => Box::new(|x, y| x + y),
                    "tensor_sub" => Box::new(|x, y| x - y),
                    "tensor_mul" => Box::new(|x, y| x * y),
                    _ => Box::new(|x, y| x / y),
                };
                match (a, b) {
                    (
                        Valor::Tensor {
                            shape: sa,
                            dados: da,
                        },
                        Valor::Tensor {
                            shape: sb,
                            dados: db,
                        },
                    ) => {
                        if sa != sb {
                            return Err(format!(
                                "{}: shapes incompativeis {:?} vs {:?}",
                                nome, sa, sb
                            ));
                        }
                        Ok(Valor::Tensor {
                            shape: sa,
                            dados: Arc::new(
                                da.iter().zip(db.iter()).map(|(x, y)| op(*x, *y)).collect(),
                            ),
                        })
                    }
                    (Valor::Tensor { shape, dados }, escalar) => {
                        let esc = to_f64(&escalar, nome)?;
                        Ok(Valor::Tensor {
                            shape,
                            dados: Arc::new(dados.iter().map(|x| op(*x, esc)).collect()),
                        })
                    }
                    (va, _) => Err(format!("'{}' requer tensor, recebeu {}", nome, va)),
                }
            }
            "tensor_potencia" => {
                let (a, exp) = arg2(args, "tensor_potencia")?;
                let e = to_f64(&exp, "tensor_potencia")?;
                match a {
                    Valor::Tensor { shape, dados } => Ok(Valor::Tensor {
                        shape,
                        dados: Arc::new(dados.iter().map(|x| x.powf(e)).collect()),
                    }),
                    v => Err(format!("'tensor_potencia' requer tensor, recebeu {}", v)),
                }
            }
            "tensor_neg" => match arg1(args, "tensor_neg")? {
                Valor::Tensor { shape, dados } => Ok(Valor::Tensor {
                    shape,
                    dados: Arc::new(dados.iter().map(|x| -x).collect()),
                }),
                v => Err(format!("'tensor_neg' requer tensor, recebeu {}", v)),
            },
            "tensor_exp" => match arg1(args, "tensor_exp")? {
                Valor::Tensor { shape, dados } => Ok(Valor::Tensor {
                    shape,
                    dados: Arc::new(dados.iter().map(|x| x.exp()).collect()),
                }),
                v => Err(format!("'tensor_exp' requer tensor, recebeu {}", v)),
            },
            "tensor_log" => match arg1(args, "tensor_log")? {
                Valor::Tensor { shape, dados } => Ok(Valor::Tensor {
                    shape,
                    dados: Arc::new(dados.iter().map(|x| x.ln()).collect()),
                }),
                v => Err(format!("'tensor_log' requer tensor, recebeu {}", v)),
            },
            "tensor_raiz" => match arg1(args, "tensor_raiz")? {
                Valor::Tensor { shape, dados } => Ok(Valor::Tensor {
                    shape,
                    dados: Arc::new(dados.iter().map(|x| x.sqrt()).collect()),
                }),
                v => Err(format!("'tensor_raiz' requer tensor, recebeu {}", v)),
            },
            "tensor_relu" => match arg1(args, "tensor_relu")? {
                Valor::Tensor { shape, dados } => Ok(Valor::Tensor {
                    shape,
                    dados: Arc::new(dados.iter().map(|x| x.max(0.0)).collect()),
                }),
                v => Err(format!("'tensor_relu' requer tensor, recebeu {}", v)),
            },
            "tensor_sigmoid" => match arg1(args, "tensor_sigmoid")? {
                Valor::Tensor { shape, dados } => Ok(Valor::Tensor {
                    shape,
                    dados: Arc::new(dados.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()),
                }),
                v => Err(format!("'tensor_sigmoid' requer tensor, recebeu {}", v)),
            },
            "tensor_tanh" => match arg1(args, "tensor_tanh")? {
                Valor::Tensor { shape, dados } => Ok(Valor::Tensor {
                    shape,
                    dados: Arc::new(dados.iter().map(|x| x.tanh()).collect()),
                }),
                v => Err(format!("'tensor_tanh' requer tensor, recebeu {}", v)),
            },
            "tensor_softmax" => match arg1(args, "tensor_softmax")? {
                Valor::Tensor { shape, dados } => {
                    let max = dados.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = dados.iter().map(|x| (x - max).exp()).collect();
                    let soma: f64 = exps.iter().sum();
                    Ok(Valor::Tensor {
                        shape,
                        dados: Arc::new(exps.into_iter().map(|e| e / soma).collect()),
                    })
                }
                v => Err(format!("'tensor_softmax' requer tensor, recebeu {}", v)),
            },
            "tensor_media" => match arg1(args, "tensor_media")? {
                Valor::Tensor { dados, .. } if !dados.is_empty() => Ok(Valor::Numero(
                    dados.iter().sum::<f64>() / dados.len() as f64,
                )),
                Valor::Tensor { .. } => Err("tensor_media: tensor vazio".to_string()),
                v => Err(format!("'tensor_media' requer tensor, recebeu {}", v)),
            },
            "tensor_soma_total" => match arg1(args, "tensor_soma_total")? {
                Valor::Tensor { dados, .. } => Ok(Valor::Numero(dados.iter().sum())),
                v => Err(format!("'tensor_soma_total' requer tensor, recebeu {}", v)),
            },
            "tensor_max" => match arg1(args, "tensor_max")? {
                Valor::Tensor { dados, .. } => Ok(Valor::Numero(
                    dados.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                )),
                v => Err(format!("'tensor_max' requer tensor, recebeu {}", v)),
            },
            "tensor_min" => match arg1(args, "tensor_min")? {
                Valor::Tensor { dados, .. } => Ok(Valor::Numero(
                    dados.iter().cloned().fold(f64::INFINITY, f64::min),
                )),
                v => Err(format!("'tensor_min' requer tensor, recebeu {}", v)),
            },

            // -- Quantização -----------------------------------------------
            "tensor_quantizar_int8" => {
                // quantizar_int8(tensor) → Mapa { dados: Bytes, escala: Numero, zero_ponto: Inteiro, shape: Lista }
                match arg1(args, "tensor_quantizar_int8")? {
                    Valor::Tensor { shape, dados } => {
                        let min = dados.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max = dados.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let escala = if (max - min).abs() < 1e-10 {
                            1.0
                        } else {
                            (max - min) / 255.0
                        };
                        let zero_ponto = (-min / escala).round() as i64;
                        let bytes: Vec<u8> = dados
                            .iter()
                            .map(|&x| {
                                ((x / escala).round() + zero_ponto as f64).clamp(0.0, 255.0) as u8
                            })
                            .collect();
                        let shape_lista: Vec<Valor> =
                            shape.iter().map(|&d| Valor::Inteiro(d as i64)).collect();
                        let mut mapa = HashMap::new();
                        mapa.insert("dados".to_string(), Valor::Bytes(Arc::new(bytes)));
                        mapa.insert("escala".to_string(), Valor::Numero(escala));
                        mapa.insert("zero_ponto".to_string(), Valor::Inteiro(zero_ponto));
                        mapa.insert("shape".to_string(), Valor::Lista(shape_lista));
                        Ok(Valor::Mapa(mapa))
                    }
                    v => Err(format!(
                        "'tensor_quantizar_int8' requer tensor, recebeu {}",
                        v
                    )),
                }
            }
            "tensor_dequantizar_int8" => {
                // dequantizar_int8(mapa) → Tensor
                match arg1(args, "tensor_dequantizar_int8")? {
                    Valor::Mapa(m) => {
                        let bytes = match m.get("dados") {
                            Some(Valor::Bytes(b)) => b.clone(),
                            _ => {
                                return Err(
                                    "tensor_dequantizar_int8: campo 'dados' ausente".to_string()
                                )
                            }
                        };
                        let escala = match m.get("escala") {
                            Some(Valor::Numero(s)) => *s,
                            _ => {
                                return Err(
                                    "tensor_dequantizar_int8: campo 'escala' ausente".to_string()
                                )
                            }
                        };
                        let zero_ponto = match m.get("zero_ponto") {
                            Some(Valor::Inteiro(z)) => *z,
                            _ => {
                                return Err("tensor_dequantizar_int8: campo 'zero_ponto' ausente"
                                    .to_string())
                            }
                        };
                        let shape = match m.get("shape") {
                            Some(Valor::Lista(s)) => s
                                .iter()
                                .map(|v| match v {
                                    Valor::Inteiro(n) => Ok(*n as usize),
                                    _ => Err("shape deve ser lista de inteiros".to_string()),
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                            _ => {
                                return Err(
                                    "tensor_dequantizar_int8: campo 'shape' ausente".to_string()
                                )
                            }
                        };
                        let dados: Vec<f64> = bytes
                            .iter()
                            .map(|&b| (b as f64 - zero_ponto as f64) * escala)
                            .collect();
                        Ok(Valor::Tensor {
                            shape,
                            dados: Arc::new(dados),
                        })
                    }
                    v => Err(format!(
                        "'tensor_dequantizar_int8' requer mapa de quantizacao, recebeu {}",
                        v
                    )),
                }
            }
            "tensor_quantizar_f16" => {
                // quantizar_f16(tensor) → Mapa { dados: Bytes (2 bytes/elem, little-endian f16), shape: Lista }
                match arg1(args, "tensor_quantizar_f16")? {
                    Valor::Tensor { shape, dados } => {
                        let mut bytes = Vec::with_capacity(dados.len() * 2);
                        for &x in dados.iter() {
                            let bits = f64_para_f16_bits(x);
                            bytes.push((bits & 0xFF) as u8);
                            bytes.push((bits >> 8) as u8);
                        }
                        let shape_lista: Vec<Valor> =
                            shape.iter().map(|&d| Valor::Inteiro(d as i64)).collect();
                        let mut mapa = HashMap::new();
                        mapa.insert("dados".to_string(), Valor::Bytes(Arc::new(bytes)));
                        mapa.insert("shape".to_string(), Valor::Lista(shape_lista));
                        Ok(Valor::Mapa(mapa))
                    }
                    v => Err(format!(
                        "'tensor_quantizar_f16' requer tensor, recebeu {}",
                        v
                    )),
                }
            }
            "tensor_dequantizar_f16" => {
                // dequantizar_f16(mapa) → Tensor
                match arg1(args, "tensor_dequantizar_f16")? {
                    Valor::Mapa(m) => {
                        let bytes = match m.get("dados") {
                            Some(Valor::Bytes(b)) => b.clone(),
                            _ => {
                                return Err(
                                    "tensor_dequantizar_f16: campo 'dados' ausente".to_string()
                                )
                            }
                        };
                        let shape = match m.get("shape") {
                            Some(Valor::Lista(s)) => s
                                .iter()
                                .map(|v| match v {
                                    Valor::Inteiro(n) => Ok(*n as usize),
                                    _ => Err("shape deve ser lista de inteiros".to_string()),
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                            _ => {
                                return Err(
                                    "tensor_dequantizar_f16: campo 'shape' ausente".to_string()
                                )
                            }
                        };
                        if bytes.len() % 2 != 0 {
                            return Err("tensor_dequantizar_f16: bytes deve ter comprimento par"
                                .to_string());
                        }
                        let dados: Vec<f64> = bytes
                            .chunks_exact(2)
                            .map(|c| f16_bits_para_f64(u16::from_le_bytes([c[0], c[1]])))
                            .collect();
                        Ok(Valor::Tensor {
                            shape,
                            dados: Arc::new(dados),
                        })
                    }
                    v => Err(format!(
                        "'tensor_dequantizar_f16' requer mapa de quantizacao, recebeu {}",
                        v
                    )),
                }
            }

            // -- Imagens -------------------------------------------------------
            "imagem_ler" => match arg1(args, "imagem_ler")? {
                // Retorna Bytes (PNG bruto decodificado como RGBA8)
                Valor::Texto(p) => {
                    let img = image::open(&p)
                        .map_err(|e| format!("imagem_ler('{}'): {}", p, e))?
                        .to_rgba8();
                    Ok(Valor::Bytes(Arc::new(img.into_raw())))
                }
                v => Err(format!("'imagem_ler' requer texto, recebeu {}", v)),
            },
            "imagem_ler_tensor" => match arg1(args, "imagem_ler_tensor")? {
                // Retorna Tensor[H, W, C] com valores normalizados 0..1
                Valor::Texto(p) => {
                    let img = image::open(&p)
                        .map_err(|e| format!("imagem_ler_tensor('{}'): {}", p, e))?
                        .to_rgb8();
                    let (w, h) = img.dimensions();
                    let dados: Vec<f64> =
                        img.into_raw().iter().map(|&b| b as f64 / 255.0).collect();
                    Ok(Valor::Tensor {
                        shape: vec![h as usize, w as usize, 3],
                        dados: Arc::new(dados),
                    })
                }
                v => Err(format!("'imagem_ler_tensor' requer texto, recebeu {}", v)),
            },
            "imagem_info" => match arg1(args, "imagem_info")? {
                Valor::Texto(p) => {
                    let img =
                        image::open(&p).map_err(|e| format!("imagem_info('{}'): {}", p, e))?;
                    let mut mapa = HashMap::new();
                    mapa.insert("largura".to_string(), Valor::Inteiro(img.width() as i64));
                    mapa.insert("altura".to_string(), Valor::Inteiro(img.height() as i64));
                    mapa.insert(
                        "canais".to_string(),
                        Valor::Inteiro(img.color().channel_count() as i64),
                    );
                    Ok(Valor::Mapa(mapa))
                }
                v => Err(format!("'imagem_info' requer texto, recebeu {}", v)),
            },
            "imagem_largura" => match arg1(args, "imagem_largura")? {
                Valor::Texto(p) => {
                    let img =
                        image::open(&p).map_err(|e| format!("imagem_largura('{}'): {}", p, e))?;
                    Ok(Valor::Inteiro(img.width() as i64))
                }
                v => Err(format!("'imagem_largura' requer texto, recebeu {}", v)),
            },
            "imagem_altura" => match arg1(args, "imagem_altura")? {
                Valor::Texto(p) => {
                    let img =
                        image::open(&p).map_err(|e| format!("imagem_altura('{}'): {}", p, e))?;
                    Ok(Valor::Inteiro(img.height() as i64))
                }
                v => Err(format!("'imagem_altura' requer texto, recebeu {}", v)),
            },
            "imagem_tensor_para_rgb" => {
                // tensor_para_rgb(tensor_HWC) → Bytes (raw RGB u8)
                // Valores esperados 0..1 (normalizados) ou 0..255
                match arg1(args, "imagem_tensor_para_rgb")? {
                    Valor::Tensor { shape, dados } => {
                        if shape.len() != 3 {
                            return Err("imagem_tensor_para_rgb: tensor deve ter shape [H, W, C]"
                                .to_string());
                        }
                        let max_val = dados.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let escala = if max_val <= 1.0 { 255.0 } else { 1.0 };
                        let bytes: Vec<u8> = dados
                            .iter()
                            .map(|&x| (x * escala).clamp(0.0, 255.0) as u8)
                            .collect();
                        Ok(Valor::Bytes(Arc::new(bytes)))
                    }
                    v => Err(format!(
                        "'imagem_tensor_para_rgb' requer tensor, recebeu {}",
                        v
                    )),
                }
            }
            "imagem_salvar" => {
                // imagem_salvar(caminho, bytes_rgb, largura, altura)
                if args.len() < 4 {
                    return Err("imagem_salvar(caminho, bytes, largura, altura)".to_string());
                }
                let caminho = match &args[0] {
                    Valor::Texto(p) => p.clone(),
                    v => {
                        return Err(format!(
                            "imagem_salvar: caminho deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let bytes = match &args[1] {
                    Valor::Bytes(b) => b.clone(),
                    v => return Err(format!("imagem_salvar: bytes invalidos, recebeu {}", v)),
                };
                let w = to_usize(&args[2], "imagem_salvar")?;
                let h = to_usize(&args[3], "imagem_salvar")?;
                let canais = bytes.len() / (w * h);
                let img: image::DynamicImage = if canais == 3 {
                    let buf = image::RgbImage::from_raw(w as u32, h as u32, bytes.to_vec())
                        .ok_or("imagem_salvar: dimensoes incorretas para RGB")?;
                    image::DynamicImage::ImageRgb8(buf)
                } else if canais == 4 {
                    let buf = image::RgbaImage::from_raw(w as u32, h as u32, bytes.to_vec())
                        .ok_or("imagem_salvar: dimensoes incorretas para RGBA")?;
                    image::DynamicImage::ImageRgba8(buf)
                } else {
                    return Err(format!(
                        "imagem_salvar: {} canais nao suportado (use 3=RGB ou 4=RGBA)",
                        canais
                    ));
                };
                img.save(&caminho)
                    .map_err(|e| format!("imagem_salvar('{}'): {}", caminho, e))?;
                Ok(Valor::Booleano(true))
            }

            // -- PDF -----------------------------------------------------------
            "pdf_informacoes" => {
                let (caminho, senha) = argumentos_pdf(args, "pdf_informacoes")?;
                let info = crate::pdf::informacoes(&caminho, senha.as_deref())?;
                let mut mapa = HashMap::new();
                mapa.insert("titulo".to_string(), opcao_texto(info.titulo));
                mapa.insert("autor".to_string(), opcao_texto(info.autor));
                mapa.insert("assunto".to_string(), opcao_texto(info.assunto));
                mapa.insert(
                    "palavras_chave".to_string(),
                    opcao_texto(info.palavras_chave),
                );
                mapa.insert("criador".to_string(), opcao_texto(info.criador));
                mapa.insert("produtor".to_string(), opcao_texto(info.produtor));
                mapa.insert("data_criacao".to_string(), opcao_texto(info.data_criacao));
                mapa.insert(
                    "data_modificacao".to_string(),
                    opcao_texto(info.data_modificacao),
                );
                mapa.insert("paginas".to_string(), Valor::Inteiro(info.paginas as i64));
                mapa.insert("versao".to_string(), Valor::Texto(info.versao));
                Ok(Valor::Mapa(mapa))
            }
            "pdf_numero_paginas" => {
                let (caminho, senha) = argumentos_pdf(args, "pdf_numero_paginas")?;
                crate::pdf::numero_paginas(&caminho, senha.as_deref())
                    .map(|n| Valor::Inteiro(n as i64))
            }
            "pdf_extrair_texto" => {
                let (caminho, senha) = argumentos_pdf(args, "pdf_extrair_texto")?;
                crate::pdf::extrair_texto(&caminho, senha.as_deref()).map(Valor::Texto)
            }
            "pdf_extrair_paginas" => {
                let (caminho, senha) = argumentos_pdf(args, "pdf_extrair_paginas")?;
                crate::pdf::extrair_paginas(&caminho, senha.as_deref())
                    .map(|paginas| Valor::Lista(paginas.into_iter().map(Valor::Texto).collect()))
            }
            "pdf_extrair_pagina" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err("pdf_extrair_pagina(caminho, pagina, senha?)".to_string());
                }
                let caminho = match &args[0] {
                    Valor::Texto(s) => s.clone(),
                    v => {
                        return Err(format!(
                            "pdf_extrair_pagina: caminho deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let pagina = match &args[1] {
                    Valor::Inteiro(n) if *n > 0 => *n as u32,
                    Valor::Numero(n) if *n >= 1.0 && n.fract() == 0.0 => *n as u32,
                    v => {
                        return Err(format!(
                            "pdf_extrair_pagina: pagina deve ser inteiro positivo, recebeu {}",
                            v
                        ))
                    }
                };
                let senha = match args.get(2) {
                    Some(Valor::Texto(s)) => Some(s.as_str()),
                    Some(v) => {
                        return Err(format!(
                            "pdf_extrair_pagina: senha deve ser texto, recebeu {}",
                            v
                        ))
                    }
                    None => None,
                };
                crate::pdf::extrair_pagina(&caminho, pagina, senha).map(Valor::Texto)
            }
            "pdf_ocr_disponivel" => {
                if !args.is_empty() {
                    return Err("pdf_ocr_disponivel() nao recebe argumentos".to_string());
                }
                let estado = crate::pdf::disponibilidade_ocr();
                let mut mapa = HashMap::new();
                mapa.insert("disponivel".to_string(), Valor::Booleano(estado.disponivel));
                mapa.insert("tesseract".to_string(), Valor::Booleano(estado.tesseract));
                mapa.insert("pdftoppm".to_string(), Valor::Booleano(estado.pdftoppm));
                Ok(Valor::Mapa(mapa))
            }
            "pdf_ocr_texto" => {
                let (caminho, opcoes) = argumentos_ocr(args, "pdf_ocr_texto")?;
                crate::pdf::ocr_texto(&caminho, &opcoes).map(Valor::Texto)
            }
            "pdf_ocr_paginas" => {
                let (caminho, opcoes) = argumentos_ocr(args, "pdf_ocr_paginas")?;
                crate::pdf::ocr_paginas(&caminho, &opcoes)
                    .map(|paginas| Valor::Lista(paginas.into_iter().map(Valor::Texto).collect()))
            }
            "pdf_extrair_texto_com_ocr" => {
                let (caminho, opcoes) = argumentos_ocr(args, "pdf_extrair_texto_com_ocr")?;
                crate::pdf::extrair_texto_com_ocr(&caminho, &opcoes).map(Valor::Texto)
            }
            // -- FFI -----------------------------------------------------------
            "ffi_permitida" => {
                if !args.is_empty() {
                    return Err("ffi_permitida() nao recebe argumentos".to_string());
                }
                Ok(Valor::Booleano(crate::ffi::permitido()))
            }
            "ffi_carregar" => {
                let caminho = match arg1(args, "ffi_carregar")? {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "ffi_carregar: caminho deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                crate::ffi::carregar(&caminho).map(|id| Valor::Inteiro(id as i64))
            }
            "ffi_fechar" => {
                let id = match arg1(args, "ffi_fechar")? {
                    Valor::Inteiro(n) if n > 0 => n as u64,
                    v => {
                        return Err(format!(
                            "ffi_fechar: identificador deve ser inteiro positivo, recebeu {}",
                            v
                        ))
                    }
                };
                Ok(Valor::Booleano(crate::ffi::fechar(id)))
            }
            "ffi_chamar" => {
                if args.len() != 3 {
                    return Err("ffi_chamar(biblioteca, simbolo, dados)".to_string());
                }
                let id = match &args[0] {
                    Valor::Inteiro(n) if *n > 0 => *n as u64,
                    v => {
                        return Err(format!(
                            "ffi_chamar: biblioteca deve ser inteiro positivo, recebeu {}",
                            v
                        ))
                    }
                };
                let simbolo = match &args[1] {
                    Valor::Texto(s) => s.clone(),
                    v => return Err(format!("ffi_chamar: simbolo deve ser texto, recebeu {}", v)),
                };
                let entrada = valor_para_json(&args[2]);
                let resposta = crate::ffi::chamar(id, &simbolo, &entrada)?;
                if resposta.trim().is_empty() {
                    Ok(Valor::Nulo)
                } else {
                    json_para_valor(resposta.trim())
                }
            }

            "tensor_soma_eixo" => {
                if args.len() < 2 {
                    return Err("'tensor_soma_eixo' requer tensor, eixo".to_string());
                }
                match args[0].clone() {
                    Valor::Tensor { shape, dados } => {
                        let eixo = to_usize(&args[1], "tensor_soma_eixo")?;
                        tensor_reducao_eixo(&shape, &dados, eixo, |acc, x| acc + x, 0.0)
                    }
                    v => Err(format!("'tensor_soma_eixo' requer tensor, recebeu {}", v)),
                }
            }
            "tensor_media_eixo" => {
                if args.len() < 2 {
                    return Err("'tensor_media_eixo' requer tensor, eixo".to_string());
                }
                match args[0].clone() {
                    Valor::Tensor { shape, dados } => {
                        let eixo = to_usize(&args[1], "tensor_media_eixo")?;
                        let n = shape.get(eixo).copied().unwrap_or(1) as f64;
                        match tensor_reducao_eixo(&shape, &dados, eixo, |acc, x| acc + x, 0.0)? {
                            Valor::Tensor {
                                shape: s2,
                                dados: d2,
                            } => Ok(Valor::Tensor {
                                shape: s2,
                                dados: Arc::new(d2.iter().map(|x| x / n).collect()),
                            }),
                            v => Ok(v),
                        }
                    }
                    v => Err(format!("'tensor_media_eixo' requer tensor, recebeu {}", v)),
                }
            }
            "tensor_concatenar" => {
                if args.len() < 2 {
                    return Err("'tensor_concatenar' requer lista_de_tensores, eixo".to_string());
                }
                let eixo = to_usize(&args[1], "tensor_concatenar")?;
                match args[0].clone() {
                    Valor::Lista(lista) => {
                        let tensores: Result<Vec<(Vec<usize>, Arc<Vec<f64>>)>, String> = lista
                            .into_iter()
                            .map(|v| match v {
                                Valor::Tensor { shape, dados } => Ok((shape, dados)),
                                o => {
                                    Err(format!("tensor_concatenar: elemento nao e tensor: {}", o))
                                }
                            })
                            .collect();
                        tensor_concat(&tensores?, eixo)
                    }
                    v => Err(format!(
                        "'tensor_concatenar' requer lista de tensores, recebeu {}",
                        v
                    )),
                }
            }

            // -- Buffer de saida ----------------------------------------------
            "capturar_saida" => {
                let funcao = arg1(args, "capturar_saida")?;
                // Salva estado atual e habilita buffer temporario
                let modo_ant = MODO_SERVIDOR.with(|m| {
                    let p = m.get();
                    m.set(true);
                    p
                });
                let saida_ant = SAIDA_HTTP.with(|s| std::mem::take(&mut *s.borrow_mut()));
                let r = self.chamar_valor("<capturar_saida>", funcao, vec![], env);
                let capturado = SAIDA_HTTP.with(|s| std::mem::take(&mut *s.borrow_mut()));
                MODO_SERVIDOR.with(|m| m.set(modo_ant));
                SAIDA_HTTP.with(|s| *s.borrow_mut() = saida_ant);
                r?;
                Ok(Valor::Texto(
                    String::from_utf8_lossy(&capturado).into_owned(),
                ))
            }

            // -- Formatacao avancada ------------------------------------------
            "formatar" => {
                let (padrao, dados) = arg2(args, "formatar")?;
                let padrao = match padrao {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'formatar' requer texto como primeiro argumento, recebeu {}",
                            v
                        ))
                    }
                };
                let mapa = match dados {
                    Valor::Mapa(m) => m,
                    v => {
                        return Err(format!(
                            "'formatar' requer mapa como segundo argumento, recebeu {}",
                            v
                        ))
                    }
                };
                let resultado = formatar_padrao(&padrao, &mapa)?;
                Ok(Valor::Texto(resultado))
            }

            // -- Regex --------------------------------------------------------
            "regex_combinar" => {
                let (padrao, texto) = arg2(args, "regex_combinar")?;
                let padrao = match padrao {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'regex_combinar' padrao deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let texto = match texto {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'regex_combinar' texto deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let re =
                    regex::Regex::new(&padrao).map_err(|e| format!("regex invalido: {}", e))?;
                match re.captures(&texto) {
                    None => Ok(Valor::Nulo),
                    Some(cap) => {
                        let grupos: Vec<Valor> = cap
                            .iter()
                            .map(|m| {
                                m.map(|s| Valor::Texto(s.as_str().to_string()))
                                    .unwrap_or(Valor::Nulo)
                            })
                            .collect();
                        Ok(Valor::Lista(grupos))
                    }
                }
            }
            "regex_combinar_tudo" => {
                let (padrao, texto) = arg2(args, "regex_combinar_tudo")?;
                let padrao = match padrao {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'regex_combinar_tudo' padrao deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let texto = match texto {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'regex_combinar_tudo' texto deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let re =
                    regex::Regex::new(&padrao).map_err(|e| format!("regex invalido: {}", e))?;
                let resultados: Vec<Valor> = re
                    .find_iter(&texto)
                    .map(|m| Valor::Texto(m.as_str().to_string()))
                    .collect();
                Ok(Valor::Lista(resultados))
            }
            "regex_substituir" => {
                if args.len() < 3 {
                    return Err(
                        "'regex_substituir' requer 3 argumentos: padrao, substituto, texto"
                            .to_string(),
                    );
                }
                let padrao = match args[0].clone() {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'regex_substituir' padrao deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let subst = match args[1].clone() {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'regex_substituir' substituto deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let texto = match args[2].clone() {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'regex_substituir' texto deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let re =
                    regex::Regex::new(&padrao).map_err(|e| format!("regex invalido: {}", e))?;
                Ok(Valor::Texto(
                    re.replace_all(&texto, subst.as_str()).into_owned(),
                ))
            }
            "regex_dividir" => {
                let (padrao, texto) = arg2(args, "regex_dividir")?;
                let padrao = match padrao {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'regex_dividir' padrao deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let texto = match texto {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'regex_dividir' texto deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let re =
                    regex::Regex::new(&padrao).map_err(|e| format!("regex invalido: {}", e))?;
                let partes: Vec<Valor> = re
                    .split(&texto)
                    .map(|s| Valor::Texto(s.to_string()))
                    .collect();
                Ok(Valor::Lista(partes))
            }

            // -- Base64 -------------------------------------------------------
            "base64_codificar" => {
                let v = arg1(args, "base64_codificar")?;
                let bytes = match &v {
                    Valor::Texto(s) => s.as_bytes().to_vec(),
                    _ => return Err(format!("'base64_codificar' requer texto, recebeu {}", v)),
                };
                Ok(Valor::Texto(B64.encode(bytes)))
            }
            "base64_decodificar" => {
                let s = match arg1(args, "base64_decodificar")? {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'base64_decodificar' requer texto, recebeu {}", v)),
                };
                let bytes = B64
                    .decode(s.trim())
                    .map_err(|e| format!("base64 invalido: {}", e))?;
                Ok(Valor::Texto(String::from_utf8_lossy(&bytes).into_owned()))
            }

            // -- Hash e HMAC --------------------------------------------------
            "sha256" => {
                let s = match arg1(args, "sha256")? {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'sha256' requer texto, recebeu {}", v)),
                };
                let mut h = Sha256::new();
                h.update(s.as_bytes());
                Ok(Valor::Texto(hex::encode(h.finalize())))
            }
            "md5" => {
                let s = match arg1(args, "md5")? {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'md5' requer texto, recebeu {}", v)),
                };
                Ok(Valor::Texto(format!("{:x}", md5_simples(s.as_bytes()))))
            }
            "hmac_sha256" => {
                let (msg_v, chave_v) = arg2(args, "hmac_sha256")?;
                let msg = match msg_v {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'hmac_sha256' mensagem deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let chave = match chave_v {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'hmac_sha256' chave deve ser texto, recebeu {}", v)),
                };
                let mut mac = Hmac::<Sha256>::new_from_slice(chave.as_bytes())
                    .map_err(|e| format!("hmac_sha256: {}", e))?;
                mac.update(msg.as_bytes());
                Ok(Valor::Texto(hex::encode(mac.finalize().into_bytes())))
            }

            // -- Erros tipados ------------------------------------------------
            "Erro" => match args.len() {
                1 => {
                    let msg = arg1(args, "Erro")?.to_string();
                    Ok(Valor::Erro {
                        tipo: "Erro".to_string(),
                        mensagem: msg,
                        pilha: self.capturar_pilha(),
                    })
                }
                2 => {
                    let (tipo_v, msg_v) = arg2(args, "Erro")?;
                    let tipo = match tipo_v {
                        Valor::Texto(s) => s,
                        v => return Err(format!("'Erro' requer tipo em texto, recebeu {}", v)),
                    };
                    Ok(Valor::Erro {
                        tipo,
                        mensagem: msg_v.to_string(),
                        pilha: self.capturar_pilha(),
                    })
                }
                n => Err(format!("'Erro' espera 1 ou 2 argumentos, recebeu {}", n)),
            },
            "tipo_erro" => match arg1(args, "tipo_erro")? {
                Valor::Erro { tipo, .. } => Ok(Valor::Texto(tipo)),
                v => Err(format!("'tipo_erro' requer um Erro, recebeu {}", v)),
            },
            "mensagem_erro" => match arg1(args, "mensagem_erro")? {
                Valor::Erro { mensagem, .. } => Ok(Valor::Texto(mensagem)),
                v => Err(format!("'mensagem_erro' requer um Erro, recebeu {}", v)),
            },
            "pilha_erro" => match arg1(args, "pilha_erro")? {
                Valor::Erro { pilha, .. } => {
                    Ok(Valor::Lista(pilha.into_iter().map(Valor::Texto).collect()))
                }
                v => Err(format!("'pilha_erro' requer um Erro, recebeu {}", v)),
            },
            "entrada_get" => {
                let chave = match arg1(args, "entrada_get")? {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'entrada_get' requer texto, recebeu {}", v)),
                };
                match std::env::var("PEP_QUERY_STRING") {
                    Ok(q) => Ok(parsear_query_valor(&q, &chave)),
                    Err(_) => Ok(Valor::Nulo),
                }
            }
            "entrada_post" => {
                let chave = match arg1(args, "entrada_post")? {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'entrada_post' requer texto, recebeu {}", v)),
                };
                match std::env::var("PEP_POST_DATA") {
                    Ok(q) => Ok(parsear_query_valor(&q, &chave)),
                    Err(_) => Ok(Valor::Nulo),
                }
            }
            "obter" => match args.as_slice() {
                [Valor::Mapa(m), Valor::Texto(k)] => Ok(m.get(k).cloned().unwrap_or(Valor::Nulo)),
                [Valor::Mapa(m), Valor::Texto(k), padrao] => {
                    Ok(m.get(k).cloned().unwrap_or_else(|| padrao.clone()))
                }
                _ => Err("obter(mapa, chave, padrao?)".to_string()),
            },

            // -- Banco de dados ------------------------------------------------
            "obter_url" => {
                let url = match arg1(args, "obter_url")? {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'obter_url' requer URL em texto, recebeu {}", v)),
                };
                let agent = agente_http();
                let mut resposta = agent
                    .get(&url)
                    .call()
                    .map_err(|e| format!("obter_url('{}'): {}", url, e))?;
                let corpo = resposta
                    .body_mut()
                    .read_to_string()
                    .map_err(|e| format!("obter_url('{}'): {}", url, e))?;
                Ok(Valor::Texto(corpo))
            }
            "postar_url" => {
                let (url, dados) = arg2(args, "postar_url")?;
                let url = match url {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'postar_url' requer URL em texto, recebeu {}", v)),
                };
                let (corpo, content_type) = match dados {
                    Valor::Texto(s) => (s, "text/plain; charset=utf-8"),
                    valor => (valor_para_json(&valor), "application/json; charset=utf-8"),
                };
                let agent = agente_http();
                let mut resposta = agent
                    .post(&url)
                    .header("Content-Type", content_type)
                    .send(corpo)
                    .map_err(|e| format!("postar_url('{}'): {}", url, e))?;
                let resposta_texto = resposta
                    .body_mut()
                    .read_to_string()
                    .map_err(|e| format!("postar_url('{}'): {}", url, e))?;
                Ok(Valor::Texto(resposta_texto))
            }

            "sqlite_conectar" => {
                let caminho = match arg1(args, "sqlite_conectar")? {
                    Valor::Texto(s) => s,
                    v => {
                        return Err(format!(
                            "'sqlite_conectar' requer caminho em texto, recebeu {}",
                            v
                        ))
                    }
                };
                let conn = rusqlite::Connection::open(&caminho)
                    .map_err(|e| format!("sqlite_conectar('{}'): {}", caminho, e))?;
                let id = self.proximo_id_bd.get();
                self.proximo_id_bd.set(id + 1);
                self.conexoes_sqlite.borrow_mut().insert(id, conn);
                Ok(Valor::ConexaoSQLite(id))
            }
            "sqlite_executar" => {
                let (conn_val, sql, params) = argumentos_sql(args, "sqlite_executar")?;
                let id = match conn_val {
                    Valor::ConexaoSQLite(id) => id,
                    v => {
                        return Err(format!(
                            "'sqlite_executar' requer conexao SQLite, recebeu {}",
                            v
                        ))
                    }
                };
                let valores: Vec<rusqlite::types::Value> =
                    params.into_iter().map(valor_para_sqlite).collect();
                let conexoes = self.conexoes_sqlite.borrow();
                let conn = conexoes
                    .get(&id)
                    .ok_or_else(|| format!("Conexao SQLite #{} nao encontrada", id))?;
                let afetadas = conn
                    .execute(&sql, rusqlite::params_from_iter(valores))
                    .map_err(|e| format!("sqlite_executar: {}", e))?;
                Ok(Valor::Inteiro(afetadas as i64))
            }
            "sqlite_consultar" => {
                let (conn_val, sql, params) = argumentos_sql(args, "sqlite_consultar")?;
                let id = match conn_val {
                    Valor::ConexaoSQLite(id) => id,
                    v => {
                        return Err(format!(
                            "'sqlite_consultar' requer conexao SQLite, recebeu {}",
                            v
                        ))
                    }
                };
                let valores: Vec<rusqlite::types::Value> =
                    params.into_iter().map(valor_para_sqlite).collect();
                let conexoes = self.conexoes_sqlite.borrow();
                let conn = conexoes
                    .get(&id)
                    .ok_or_else(|| format!("Conexao SQLite #{} nao encontrada", id))?;
                let mut stmt = conn
                    .prepare(&sql)
                    .map_err(|e| format!("sqlite_consultar: {}", e))?;
                let colunas: Vec<String> =
                    stmt.column_names().iter().map(|s| s.to_string()).collect();
                let mut rows = stmt
                    .query(rusqlite::params_from_iter(valores))
                    .map_err(|e| format!("sqlite_consultar: {}", e))?;
                let mut resultado = Vec::new();
                while let Some(row) = rows
                    .next()
                    .map_err(|e| format!("sqlite_consultar: {}", e))?
                {
                    let mut mapa = HashMap::new();
                    for (i, nome) in colunas.iter().enumerate() {
                        let valor = row
                            .get_ref(i)
                            .map(sqlite_para_valor)
                            .map_err(|e| format!("sqlite_consultar: {}", e))?;
                        mapa.insert(nome.clone(), valor);
                    }
                    resultado.push(Valor::Mapa(mapa));
                }
                Ok(Valor::Lista(resultado))
            }
            "sqlite_fechar" => {
                let id = match arg1(args, "sqlite_fechar")? {
                    Valor::ConexaoSQLite(id) => id,
                    v => {
                        return Err(format!(
                            "'sqlite_fechar' requer conexao SQLite, recebeu {}",
                            v
                        ))
                    }
                };
                self.conexoes_sqlite.borrow_mut().remove(&id);
                Ok(Valor::Nulo)
            }

            "bd_conectar" => {
                let url = match arg1(args, "bd_conectar")? {
                    Valor::Texto(s) => s,
                    v => return Err(format!("'bd_conectar' requer URL em texto, recebeu {}", v)),
                };
                let opts =
                    mysql::Opts::from_url(&url).map_err(|e| format!("URL invalida: {}", e))?;
                let conn =
                    mysql::Conn::new(opts).map_err(|e| format!("Erro ao conectar: {}", e))?;
                let id = self.proximo_id_bd.get();
                self.proximo_id_bd.set(id + 1);
                self.conexoes_bd.borrow_mut().insert(id, conn);
                Ok(Valor::ConexaoBD(id))
            }
            "bd_consultar" => {
                let (conn_val, sql, params) = argumentos_sql(args, "bd_consultar")?;
                let id = match conn_val {
                    Valor::ConexaoBD(id) => id,
                    v => return Err(format!("'bd_consultar' requer conexao, recebeu {}", v)),
                };
                let mysql_params =
                    mysql::Params::Positional(params.into_iter().map(valor_para_mysql).collect());
                let mut conexoes = self.conexoes_bd.borrow_mut();
                let conn = conexoes
                    .get_mut(&id)
                    .ok_or_else(|| format!("Conexao BD #{} nao encontrada", id))?;
                let linhas = conn
                    .exec_map(&sql, mysql_params, |row: mysql::Row| {
                        let mut mapa = HashMap::new();
                        for (col, val) in row.columns().iter().zip(row.unwrap()) {
                            mapa.insert(col.name_str().to_string(), mysql_val_para_pep(val));
                        }
                        Valor::Mapa(mapa)
                    })
                    .map_err(|e| format!("Erro na consulta: {}", e))?;
                Ok(Valor::Lista(linhas))
            }
            "bd_executar" => {
                let (conn_val, sql, params) = argumentos_sql(args, "bd_executar")?;
                let id = match conn_val {
                    Valor::ConexaoBD(id) => id,
                    v => return Err(format!("'bd_executar' requer conexao, recebeu {}", v)),
                };
                let mysql_params =
                    mysql::Params::Positional(params.into_iter().map(valor_para_mysql).collect());
                let mut conexoes = self.conexoes_bd.borrow_mut();
                let conn = conexoes
                    .get_mut(&id)
                    .ok_or_else(|| format!("Conexao BD #{} nao encontrada", id))?;
                conn.exec_drop(&sql, mysql_params)
                    .map_err(|e| format!("Erro ao executar: {}", e))?;
                Ok(Valor::Inteiro(conn.affected_rows() as i64))
            }
            "bd_fechar" => {
                let id = match arg1(args, "bd_fechar")? {
                    Valor::ConexaoBD(id) => id,
                    v => return Err(format!("'bd_fechar' requer conexao, recebeu {}", v)),
                };
                self.conexoes_bd.borrow_mut().remove(&id);
                Ok(Valor::Nulo)
            }

            // ── Bytes ───────────────────────────────────────────────────────
            "bytes_de_texto" => match arg1(args, "bytes_de_texto")? {
                Valor::Texto(s) => Ok(Valor::Bytes(Arc::new(s.as_bytes().to_vec()))),
                v => Err(format!("'bytes_de_texto' requer texto, recebeu {}", v)),
            },
            "bytes_para_texto" => match arg1(args, "bytes_para_texto")? {
                Valor::Bytes(b) => String::from_utf8(b.to_vec())
                    .map(Valor::Texto)
                    .map_err(|_| "bytes_para_texto: bytes nao sao UTF-8 valido".to_string()),
                v => Err(format!("'bytes_para_texto' requer bytes, recebeu {}", v)),
            },
            "bytes_tamanho" => match arg1(args, "bytes_tamanho")? {
                Valor::Bytes(b) => Ok(Valor::Inteiro(b.len() as i64)),
                v => Err(format!("'bytes_tamanho' requer bytes, recebeu {}", v)),
            },
            "bytes_fatia" => {
                if args.len() < 3 {
                    return Err("bytes_fatia(bytes, ini, fim)".to_string());
                }
                match &args[0] {
                    Valor::Bytes(b) => {
                        let ini = to_usize(&args[1], "bytes_fatia")?;
                        let fim = to_usize(&args[2], "bytes_fatia")?;
                        let fim = fim.min(b.len());
                        let ini = ini.min(fim);
                        Ok(Valor::Bytes(Arc::new(b[ini..fim].to_vec())))
                    }
                    v => Err(format!("'bytes_fatia' requer bytes, recebeu {}", v)),
                }
            }
            "bytes_obter" => {
                if args.len() < 2 {
                    return Err("bytes_obter(bytes, indice)".to_string());
                }
                match &args[0] {
                    Valor::Bytes(b) => {
                        let i = to_usize(&args[1], "bytes_obter")?;
                        b.get(i)
                            .map(|&x| Valor::Inteiro(x as i64))
                            .ok_or_else(|| format!("bytes_obter: indice {} fora dos limites", i))
                    }
                    v => Err(format!("'bytes_obter' requer bytes, recebeu {}", v)),
                }
            }
            "bytes_de_lista" => match arg1(args, "bytes_de_lista")? {
                Valor::Lista(l) => {
                    let b: Result<Vec<u8>, String> = l
                        .iter()
                        .map(|v| match v {
                            Valor::Inteiro(n) if *n >= 0 && *n <= 255 => Ok(*n as u8),
                            Valor::Numero(n) if *n >= 0.0 && *n <= 255.0 => Ok(*n as u8),
                            _ => Err(format!(
                                "bytes_de_lista: cada elemento deve ser 0-255, recebeu {}",
                                v
                            )),
                        })
                        .collect();
                    Ok(Valor::Bytes(Arc::new(b?)))
                }
                v => Err(format!("'bytes_de_lista' requer lista, recebeu {}", v)),
            },
            "bytes_para_lista" => match arg1(args, "bytes_para_lista")? {
                Valor::Bytes(b) => Ok(Valor::Lista(
                    b.iter().map(|&x| Valor::Inteiro(x as i64)).collect(),
                )),
                v => Err(format!("'bytes_para_lista' requer bytes, recebeu {}", v)),
            },
            "bytes_base64" => match arg1(args, "bytes_base64")? {
                Valor::Bytes(b) => Ok(Valor::Texto(B64.encode(b.as_ref()))),
                Valor::Texto(s) => Ok(Valor::Texto(B64.encode(s.as_bytes()))),
                v => Err(format!(
                    "'bytes_base64' requer bytes ou texto, recebeu {}",
                    v
                )),
            },
            "bytes_para_hex" => match arg1(args, "bytes_para_hex")? {
                Valor::Bytes(b) => Ok(Valor::Texto(hex::encode(b.as_ref()))),
                v => Err(format!("'bytes_para_hex' requer bytes, recebeu {}", v)),
            },
            "bytes_concatenar" => match arg1(args, "bytes_concatenar")? {
                Valor::Lista(l) => {
                    let mut out: Vec<u8> = Vec::new();
                    for v in l {
                        match v {
                            Valor::Bytes(b) => out.extend_from_slice(b.as_ref()),
                            o => {
                                return Err(format!(
                                    "bytes_concatenar: elemento nao e bytes: {}",
                                    o
                                ))
                            }
                        }
                    }
                    Ok(Valor::Bytes(Arc::new(out)))
                }
                v => Err(format!(
                    "'bytes_concatenar' requer lista de bytes, recebeu {}",
                    v
                )),
            },
            "corpo_bruto" => {
                // Retorna _CORPO_BRUTO do ambiente (injetado pelo servidor)
                Ok(self
                    .ambiente
                    .obter("_CORPO_BRUTO")
                    .unwrap_or(Valor::Bytes(Arc::new(Vec::new()))))
            }

            // ── Modelos globais ──────────────────────────────────────────────
            "modelo_carregar" => {
                if args.len() < 2 {
                    return Err("modelo_carregar(nome, caminho_ou_valor)".to_string());
                }
                let nome = match &args[0] {
                    Valor::Texto(s) => s.clone(),
                    v => {
                        return Err(format!(
                            "modelo_carregar: nome deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                // Aceita caminho de arquivo (.pep que retorna um tensor) ou valor direto
                let valor = match &args[1] {
                    Valor::Texto(caminho) => {
                        // Carrega arquivo binário de pesos (flat f64 little-endian)
                        let dados = std::fs::read(caminho).map_err(|e| {
                            format!("modelo_carregar: nao foi possivel ler '{}': {}", caminho, e)
                        })?;
                        Valor::Bytes(Arc::new(dados))
                    }
                    outro => outro.clone(),
                };
                crate::servidor::modelo_definir(nome, valor);
                Ok(Valor::Nulo)
            }
            "modelo_obter" => {
                let nome = match arg1(args, "modelo_obter")? {
                    Valor::Texto(s) => s.clone(),
                    v => return Err(format!("modelo_obter: nome deve ser texto, recebeu {}", v)),
                };
                crate::servidor::modelo_obter(&nome)
                    .ok_or_else(|| format!("modelo_obter: modelo '{}' nao carregado", nome))
            }
            "modelo_existe" => {
                let nome = match arg1(args, "modelo_existe")? {
                    Valor::Texto(s) => s.clone(),
                    v => return Err(format!("modelo_existe: nome deve ser texto, recebeu {}", v)),
                };
                Ok(Valor::Booleano(
                    crate::servidor::modelo_obter(&nome).is_some(),
                ))
            }
            "modelo_listar" => Ok(Valor::Lista(
                crate::servidor::modelos_listar()
                    .into_iter()
                    .map(Valor::Texto)
                    .collect(),
            )),
            "modelo_descarregar" => {
                let nome = match arg1(args, "modelo_descarregar")? {
                    Valor::Texto(s) => s.clone(),
                    v => {
                        return Err(format!(
                            "modelo_descarregar: nome deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                crate::servidor::modelo_remover(&nome);
                Ok(Valor::Nulo)
            }

            // ── WebSocket ───────────────────────────────────────────────────
            "ws_aceitar" => crate::websocket::ws_aceitar().map(|id| Valor::Inteiro(id as i64)),
            "ws_receber" => match crate::websocket::ws_receber()? {
                None => Ok(Valor::Nulo),
                Some(crate::websocket::WsMensagem::Texto(s)) => Ok(Valor::Texto(s)),
                Some(crate::websocket::WsMensagem::Binario(b)) => Ok(Valor::Bytes(Arc::new(b))),
                Some(crate::websocket::WsMensagem::Fechado) => Ok(Valor::Nulo),
            },
            "ws_enviar" => match arg1(args, "ws_enviar")? {
                Valor::Texto(s) => crate::websocket::ws_enviar(&s).map(|_| Valor::Nulo),
                v => Err(format!("ws_enviar: requer texto, recebeu {}", v)),
            },
            "ws_enviar_bytes" => match arg1(args, "ws_enviar_bytes")? {
                Valor::Bytes(b) => {
                    crate::websocket::ws_enviar_bytes(b.as_ref()).map(|_| Valor::Nulo)
                }
                v => Err(format!("ws_enviar_bytes: requer bytes, recebeu {}", v)),
            },
            "ws_fechar" => {
                crate::websocket::ws_fechar();
                Ok(Valor::Nulo)
            }
            "ws_id" => Ok(crate::websocket::ws_id()
                .map(|id| Valor::Inteiro(id as i64))
                .unwrap_or(Valor::Nulo)),
            "ws_conexoes" => Ok(Valor::Lista(
                crate::websocket::ws_conexoes()
                    .into_iter()
                    .map(|id| Valor::Inteiro(id as i64))
                    .collect(),
            )),
            "ws_enviar_para" => {
                if args.len() < 2 {
                    return Err("ws_enviar_para(id, msg)".to_string());
                }
                let id = match &args[0] {
                    Valor::Inteiro(n) => *n as u64,
                    v => {
                        return Err(format!(
                            "ws_enviar_para: id deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                match &args[1] {
                    Valor::Texto(s) => crate::websocket::ws_enviar_para(id, s).map(|_| Valor::Nulo),
                    v => Err(format!("ws_enviar_para: msg deve ser texto, recebeu {}", v)),
                }
            }
            "ws_broadcast" => match arg1(args, "ws_broadcast")? {
                Valor::Texto(s) => {
                    crate::websocket::ws_broadcast(&s);
                    Ok(Valor::Nulo)
                }
                v => Err(format!("ws_broadcast: requer texto, recebeu {}", v)),
            },

            // ── mmap ────────────────────────────────────────────────────────
            "mmap_abrir" => {
                let caminho = match arg1(args, "mmap_abrir")? {
                    Valor::Texto(s) => s.clone(),
                    v => return Err(format!("mmap_abrir: caminho deve ser texto, recebeu {}", v)),
                };
                crate::mmap::mmap_abrir(&caminho).map(|id| Valor::Inteiro(id as i64))
            }
            "mmap_fechar" => {
                let id = match arg1(args, "mmap_fechar")? {
                    Valor::Inteiro(n) => n as u64,
                    v => return Err(format!("mmap_fechar: requer inteiro, recebeu {}", v)),
                };
                crate::mmap::mmap_fechar(id);
                Ok(Valor::Nulo)
            }
            "mmap_tamanho" => {
                let id = match arg1(args, "mmap_tamanho")? {
                    Valor::Inteiro(n) => n as u64,
                    v => return Err(format!("mmap_tamanho: requer inteiro, recebeu {}", v)),
                };
                crate::mmap::mmap_tamanho(id).map(|n| Valor::Inteiro(n as i64))
            }
            "mmap_ler_f32" => {
                if args.len() < 3 {
                    return Err("mmap_ler_f32(id, offset, count)".to_string());
                }
                let id = match &args[0] {
                    Valor::Inteiro(n) => *n as u64,
                    v => return Err(format!("mmap_ler_f32: id deve ser inteiro, recebeu {}", v)),
                };
                let off = match &args[1] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_ler_f32: offset deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                let cnt = match &args[2] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_ler_f32: count deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                crate::mmap::mmap_ler_f32(id, off, cnt)
            }
            "mmap_ler_f64" => {
                if args.len() < 3 {
                    return Err("mmap_ler_f64(id, offset, count)".to_string());
                }
                let id = match &args[0] {
                    Valor::Inteiro(n) => *n as u64,
                    v => return Err(format!("mmap_ler_f64: id deve ser inteiro, recebeu {}", v)),
                };
                let off = match &args[1] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_ler_f64: offset deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                let cnt = match &args[2] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_ler_f64: count deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                crate::mmap::mmap_ler_f64(id, off, cnt)
            }
            "mmap_tensor_f32" => {
                if args.len() < 4 {
                    return Err("mmap_tensor_f32(id, offset, linhas, colunas)".to_string());
                }
                let id = match &args[0] {
                    Valor::Inteiro(n) => *n as u64,
                    v => {
                        return Err(format!(
                            "mmap_tensor_f32: id deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                let off = match &args[1] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_tensor_f32: offset deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                let lin = match &args[2] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_tensor_f32: linhas deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                let col = match &args[3] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_tensor_f32: colunas deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                crate::mmap::mmap_tensor_f32(id, off, lin, col)
            }
            "mmap_tensor_f64" => {
                if args.len() < 4 {
                    return Err("mmap_tensor_f64(id, offset, linhas, colunas)".to_string());
                }
                let id = match &args[0] {
                    Valor::Inteiro(n) => *n as u64,
                    v => {
                        return Err(format!(
                            "mmap_tensor_f64: id deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                let off = match &args[1] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_tensor_f64: offset deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                let lin = match &args[2] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_tensor_f64: linhas deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                let col = match &args[3] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_tensor_f64: colunas deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                crate::mmap::mmap_tensor_f64(id, off, lin, col)
            }
            "mmap_ler_bytes" => {
                if args.len() < 3 {
                    return Err("mmap_ler_bytes(id, offset, count)".to_string());
                }
                let id = match &args[0] {
                    Valor::Inteiro(n) => *n as u64,
                    v => {
                        return Err(format!(
                            "mmap_ler_bytes: id deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                let off = match &args[1] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_ler_bytes: offset deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                let cnt = match &args[2] {
                    Valor::Inteiro(n) => *n as usize,
                    v => {
                        return Err(format!(
                            "mmap_ler_bytes: count deve ser inteiro, recebeu {}",
                            v
                        ))
                    }
                };
                crate::mmap::mmap_ler_bytes(id, off, cnt)
            }

            _ => Err(format!("Funcao '{}' nao encontrada", nome)),
        }
    }
}

// -- Helpers de argumento ------------------------------------------------------

fn localizar_modulo(base: PathBuf) -> Option<PathBuf> {
    let candidatos = [
        base.clone(),
        base.with_extension("pep"),
        base.join("mod.pep"),
        base.join("index.pep"),
    ];
    candidatos.into_iter().find(|p| p.is_file())
}

fn agente_http() -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(30)))
        .build();
    config.into()
}

fn numero_f64(v: Valor) -> Option<f64> {
    match v {
        Valor::Inteiro(n) => Some(n as f64),
        Valor::Numero(n) => Some(n),
        _ => None,
    }
}

fn argumentos_sql(mut args: Vec<Valor>, nome: &str) -> Result<(Valor, String, Vec<Valor>), String> {
    if args.len() < 2 || args.len() > 3 {
        return Err(format!("{}(conexao, sql, parametros?)", nome));
    }
    let conn = args.remove(0);
    let sql = match args.remove(0) {
        Valor::Texto(s) => s,
        v => return Err(format!("'{}' requer SQL em texto, recebeu {}", nome, v)),
    };
    let parametros = if args.is_empty() {
        Vec::new()
    } else {
        match args.remove(0) {
            Valor::Lista(v) => v,
            v => {
                return Err(format!(
                    "'{}' requer parametros em lista, recebeu {}",
                    nome, v
                ))
            }
        }
    };
    Ok((conn, sql, parametros))
}

fn valor_para_sqlite(v: Valor) -> rusqlite::types::Value {
    match v {
        Valor::Nulo => rusqlite::types::Value::Null,
        Valor::Inteiro(n) => rusqlite::types::Value::Integer(n),
        Valor::Numero(n) => rusqlite::types::Value::Real(n),
        Valor::Texto(s) => rusqlite::types::Value::Text(s),
        Valor::Booleano(b) => rusqlite::types::Value::Integer(if b { 1 } else { 0 }),
        other => rusqlite::types::Value::Text(other.to_string()),
    }
}

fn sqlite_para_valor(v: rusqlite::types::ValueRef<'_>) -> Valor {
    match v {
        rusqlite::types::ValueRef::Null => Valor::Nulo,
        rusqlite::types::ValueRef::Integer(n) => Valor::Inteiro(n),
        rusqlite::types::ValueRef::Real(n) => Valor::Numero(n),
        rusqlite::types::ValueRef::Text(s) => Valor::Texto(String::from_utf8_lossy(s).into_owned()),
        rusqlite::types::ValueRef::Blob(b) => Valor::Texto(String::from_utf8_lossy(b).into_owned()),
    }
}

fn arg_numero(v: Valor, nome: &str) -> Result<f64, String> {
    let recebido = v.to_string();
    numero_f64(v).ok_or_else(|| format!("'{}' requer numero, recebeu {}", nome, recebido))
}

fn formatar_numero_br(numero: f64, casas: usize) -> String {
    let negativo = numero.is_sign_negative();
    let absoluto = numero.abs();
    let base = format!("{:.*}", casas, absoluto);
    let (inteiro, decimal) = base.split_once('.').unwrap_or((&base, ""));
    let mut agrupado = String::new();
    for (i, c) in inteiro.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            agrupado.push('.');
        }
        agrupado.push(c);
    }
    let inteiro_br: String = agrupado.chars().rev().collect();
    let sinal = if negativo { "-" } else { "" };
    if casas == 0 {
        format!("{}{}", sinal, inteiro_br)
    } else {
        format!("{}{},{}", sinal, inteiro_br, decimal)
    }
}

fn formatar_data_simples(data: &str, formato: &str) -> Result<String, String> {
    let parte = data.split_whitespace().next().unwrap_or(data);
    let mut itens = parte.split('-');
    let ano = itens
        .next()
        .ok_or_else(|| "Data invalida; use aaaa-mm-dd".to_string())?;
    let mes = itens
        .next()
        .ok_or_else(|| "Data invalida; use aaaa-mm-dd".to_string())?;
    let dia = itens
        .next()
        .ok_or_else(|| "Data invalida; use aaaa-mm-dd".to_string())?;
    if itens.next().is_some() || ano.len() != 4 || mes.len() != 2 || dia.len() != 2 {
        return Err("Data invalida; use aaaa-mm-dd".to_string());
    }
    Ok(formato
        .replace("aaaa", ano)
        .replace("mm", mes)
        .replace("dd", dia))
}

fn numeros_f64(a: Valor, b: Valor, op: &str) -> Result<(f64, f64), String> {
    let texto_a = a.to_string();
    let texto_b = b.to_string();
    match (numero_f64(a), numero_f64(b)) {
        (Some(x), Some(y)) => Ok((x, y)),
        _ => Err(format!(
            "O operador '{}' requer numeros, recebeu {} e {}",
            op, texto_a, texto_b
        )),
    }
}

fn inteiros_i64(a: Valor, b: Valor, op: &str) -> Result<(i64, i64), String> {
    match (a, b) {
        (Valor::Inteiro(x), Valor::Inteiro(y)) => Ok((x, y)),
        (a, b) => Err(format!(
            "O operador '{}' requer inteiros, recebeu {} e {}",
            op, a, b
        )),
    }
}

fn opcao_texto(valor: Option<String>) -> Valor {
    valor.map(Valor::Texto).unwrap_or(Valor::Nulo)
}

fn argumentos_pdf(mut args: Vec<Valor>, nome: &str) -> Result<(String, Option<String>), String> {
    if args.is_empty() || args.len() > 2 {
        return Err(format!("{}(caminho, senha?)", nome));
    }
    let caminho = match args.remove(0) {
        Valor::Texto(s) => s,
        v => return Err(format!("{}: caminho deve ser texto, recebeu {}", nome, v)),
    };
    let senha = if args.is_empty() {
        None
    } else {
        match args.remove(0) {
            Valor::Texto(s) => Some(s),
            v => return Err(format!("{}: senha deve ser texto, recebeu {}", nome, v)),
        }
    };
    Ok((caminho, senha))
}

fn argumentos_ocr(
    mut args: Vec<Valor>,
    nome: &str,
) -> Result<(String, crate::pdf::OpcoesOcr), String> {
    if args.is_empty() || args.len() > 2 {
        return Err(format!("{}(caminho, opcoes?)", nome));
    }
    let caminho = match args.remove(0) {
        Valor::Texto(s) => s,
        v => return Err(format!("{}: caminho deve ser texto, recebeu {}", nome, v)),
    };
    let mut opcoes = crate::pdf::OpcoesOcr::default();
    if let Some(valor) = args.pop() {
        let mapa = match valor {
            Valor::Mapa(m) => m,
            v => return Err(format!("{}: opcoes deve ser mapa, recebeu {}", nome, v)),
        };
        if let Some(valor) = mapa.get("idioma") {
            match valor {
                Valor::Texto(s) => opcoes.idioma = s.clone(),
                v => return Err(format!("{}: idioma deve ser texto, recebeu {}", nome, v)),
            }
        }
        if let Some(valor) = mapa.get("dpi") {
            opcoes.dpi = match valor {
                Valor::Inteiro(n) if *n >= 0 => *n as u32,
                Valor::Numero(n) if *n >= 0.0 && n.fract() == 0.0 => *n as u32,
                v => return Err(format!("{}: dpi deve ser inteiro, recebeu {}", nome, v)),
            };
        }
        if let Some(valor) = mapa.get("psm") {
            opcoes.psm = match valor {
                Valor::Inteiro(n) if *n >= 0 && *n <= u8::MAX as i64 => *n as u8,
                Valor::Numero(n) if *n >= 0.0 && *n <= u8::MAX as f64 && n.fract() == 0.0 => {
                    *n as u8
                }
                v => return Err(format!("{}: psm deve ser inteiro, recebeu {}", nome, v)),
            };
        }
        if let Some(valor) = mapa.get("senha") {
            match valor {
                Valor::Texto(s) => opcoes.senha = Some(s.clone()),
                Valor::Nulo => opcoes.senha = None,
                v => return Err(format!("{}: senha deve ser texto, recebeu {}", nome, v)),
            }
        }
    }
    Ok((caminho, opcoes))
}

fn arg1(mut args: Vec<Valor>, nome: &str) -> Result<Valor, String> {
    if args.len() != 1 {
        return Err(format!(
            "'{}' requer 1 argumento, recebeu {}",
            nome,
            args.len()
        ));
    }
    Ok(args.remove(0))
}

fn arg2(mut args: Vec<Valor>, nome: &str) -> Result<(Valor, Valor), String> {
    if args.len() != 2 {
        return Err(format!(
            "'{}' requer 2 argumentos, recebeu {}",
            nome,
            args.len()
        ));
    }
    let b = args.remove(1);
    let a = args.remove(0);
    Ok((a, b))
}

fn arg3(mut args: Vec<Valor>, nome: &str) -> Result<(Valor, Valor, Valor), String> {
    if args.len() != 3 {
        return Err(format!(
            "'{}' requer 3 argumentos, recebeu {}",
            nome,
            args.len()
        ));
    }
    let c = args.remove(2);
    let b = args.remove(1);
    let a = args.remove(0);
    Ok((a, b, c))
}

fn parsear_query_valor(query: &str, chave: &str) -> Valor {
    for par in query.split('&') {
        if par.is_empty() {
            continue;
        }
        let (k, v) = par.split_once('=').unwrap_or((par, ""));
        if url_decodificar_simples(k) == chave {
            return Valor::Texto(url_decodificar_simples(v));
        }
    }
    Valor::Nulo
}

// -- Gerador de numeros pseudo-aleatorios (Xorshift64) ------------------------

fn eh_bissexto(ano: u64) -> bool {
    (ano % 4 == 0 && ano % 100 != 0) || ano % 400 == 0
}

fn timestamp_para_data(secs: u64) -> (u64, u8, u8) {
    let mut dias = secs / 86400;
    let mut ano = 1970u64;
    loop {
        let dias_no_ano = if eh_bissexto(ano) { 366 } else { 365 };
        if dias < dias_no_ano {
            break;
        }
        dias -= dias_no_ano;
        ano += 1;
    }
    let dias_por_mes: [u64; 12] = [
        31,
        if eh_bissexto(ano) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut mes = 1u8;
    for &d in &dias_por_mes {
        if dias < d {
            break;
        }
        dias -= d;
        mes += 1;
    }
    (ano, mes, (dias + 1) as u8)
}

fn id_sessao_valido(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_hexdigit() || c == '-' || c == '_')
}

fn novo_id_sessao() -> String {
    bytes_aleatorios_hex(32)
        .unwrap_or_else(|_| format!("{:016x}{:016x}", aleatorio_u64(), aleatorio_u64()))
}

fn bytes_aleatorios_hex(tamanho: usize) -> Result<String, String> {
    let mut bytes = vec![0u8; tamanho];
    getrandom::fill(&mut bytes).map_err(|e| format!("gerador aleatorio do sistema: {}", e))?;
    Ok(hex::encode(bytes))
}

static SEED: AtomicU64 = AtomicU64::new(0);

fn aleatorio_u64() -> u64 {
    let mut s = SEED.load(Ordering::Relaxed);
    if s == 0 {
        use std::time::{SystemTime, UNIX_EPOCH};
        s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        s ^= 0xdeadbeefcafe1234;
    }
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    SEED.store(s, Ordering::Relaxed);
    s
}

fn aleatorio_f64() -> f64 {
    (aleatorio_u64() >> 11) as f64 / (1u64 << 53) as f64
}

// -- MySQL -> Valor -------------------------------------------------------------

fn valor_para_mysql(v: Valor) -> mysql::Value {
    match v {
        Valor::Nulo => mysql::Value::NULL,
        Valor::Inteiro(n) => mysql::Value::Int(n),
        Valor::Numero(n) => mysql::Value::Double(n),
        Valor::Texto(s) => mysql::Value::Bytes(s.into_bytes()),
        Valor::Booleano(b) => mysql::Value::Int(if b { 1 } else { 0 }),
        other => mysql::Value::Bytes(other.to_string().into_bytes()),
    }
}

fn mysql_val_para_pep(val: mysql::Value) -> Valor {
    match val {
        mysql::Value::NULL => Valor::Nulo,
        mysql::Value::Int(i) => Valor::Inteiro(i),
        mysql::Value::UInt(u) if u <= i64::MAX as u64 => Valor::Inteiro(u as i64),
        mysql::Value::UInt(u) => Valor::Numero(u as f64),
        mysql::Value::Float(f) => Valor::Numero(f as f64),
        mysql::Value::Double(d) => Valor::Numero(d),
        mysql::Value::Bytes(b) => Valor::Texto(String::from_utf8_lossy(&b).into_owned()),
        mysql::Value::Date(a, me, d, h, m, s, _) => Valor::Texto(format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            a, me, d, h, m, s
        )),
        mysql::Value::Time(neg, dias, h, m, s, _) => Valor::Texto(format!(
            "{}{}:{:02}:{:02}",
            if neg { "-" } else { "" },
            dias * 24 + h as u32,
            m,
            s
        )),
    }
}

// -- Float16 helpers -----------------------------------------------------------

fn f64_para_f16_bits(x: f64) -> u16 {
    let x = x as f32;
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0xFF {
        // inf ou NaN
        return (sign << 15) | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let exp16 = exp - 127 + 15;
    if exp16 >= 31 {
        return (sign << 15) | 0x7C00; // overflow → inf
    }
    if exp16 <= 0 {
        // subnormal ou zero
        if exp16 < -10 {
            return sign << 15;
        }
        let m = (mant | 0x800000) >> (1 - exp16);
        return (sign << 15) | ((m >> 13) as u16);
    }
    (sign << 15) | ((exp16 as u16) << 10) | ((mant >> 13) as u16)
}

fn f16_bits_para_f64(bits: u16) -> f64 {
    let sign: u32 = ((bits >> 15) & 1) as u32;
    let exp: u32 = ((bits >> 10) & 0x1F) as u32;
    let mant: u32 = (bits & 0x3FF) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = 0u32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            (sign << 31) | ((127 - 14 + 1 - e) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7F800000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits) as f64
}

// -- CSV -----------------------------------------------------------------------

fn csv_parsear_texto(s: &str) -> Vec<Valor> {
    let mut linhas = Vec::new();
    let s = s.strip_prefix('\u{FEFF}').unwrap_or(s); // strip BOM
    let chars: Vec<char> = s.chars().collect();
    let mut pos = 0;
    while pos <= chars.len() {
        let mut linha: Vec<Valor> = Vec::new();
        loop {
            let campo = if pos < chars.len() && chars[pos] == '"' {
                // campo entre aspas — pode conter vírgulas e \n
                pos += 1;
                let mut buf = String::new();
                while pos < chars.len() {
                    if chars[pos] == '"' {
                        if pos + 1 < chars.len() && chars[pos + 1] == '"' {
                            buf.push('"');
                            pos += 2;
                        } else {
                            pos += 1;
                            break;
                        }
                    } else {
                        buf.push(chars[pos]);
                        pos += 1;
                    }
                }
                buf
            } else {
                let mut buf = String::new();
                while pos < chars.len()
                    && chars[pos] != ','
                    && chars[pos] != '\n'
                    && chars[pos] != '\r'
                {
                    buf.push(chars[pos]);
                    pos += 1;
                }
                buf
            };
            linha.push(Valor::Texto(campo));
            if pos >= chars.len() || chars[pos] == '\n' || chars[pos] == '\r' {
                break;
            }
            pos += 1; // vírgula
        }
        // avança fim de linha
        if pos < chars.len() && chars[pos] == '\r' {
            pos += 1;
        }
        if pos < chars.len() && chars[pos] == '\n' {
            pos += 1;
        }
        // ignora linha vazia no final
        if pos >= chars.len() && linha.len() == 1 {
            if let Valor::Texto(s) = &linha[0] {
                if s.is_empty() {
                    break;
                }
            }
        }
        linhas.push(Valor::Lista(linha));
        if pos >= chars.len() {
            break;
        }
    }
    linhas
}

fn csv_campo_serializar(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn csv_serializar_linhas(linhas: &[Valor]) -> String {
    let mut out = String::new();
    for linha in linhas {
        if let Valor::Lista(cols) = linha {
            let campos: Vec<String> = cols
                .iter()
                .map(|c| csv_campo_serializar(&c.to_string()))
                .collect();
            out.push_str(&campos.join(","));
        } else {
            out.push_str(&csv_campo_serializar(&linha.to_string()));
        }
        out.push('\n');
    }
    out
}

// -- JSON ----------------------------------------------------------------------

fn valor_para_json(v: &Valor) -> String {
    match v {
        Valor::Nulo => "null".to_string(),
        Valor::Booleano(true) => "true".to_string(),
        Valor::Booleano(false) => "false".to_string(),
        Valor::Inteiro(n) => n.to_string(),
        Valor::Numero(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        }
        Valor::Texto(s) => {
            let e = s
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            format!("\"{}\"", e)
        }
        Valor::Lista(v) => {
            let items: Vec<String> = v.iter().map(valor_para_json).collect();
            format!("[{}]", items.join(","))
        }
        Valor::Mapa(m) => {
            let mut pares: Vec<String> = m
                .iter()
                .map(|(k, v)| {
                    let ke = k.replace('\\', "\\\\").replace('"', "\\\"");
                    format!("\"{}\":{}", ke, valor_para_json(v))
                })
                .collect();
            pares.sort();
            format!("{{{}}}", pares.join(","))
        }
        _ => "null".to_string(),
    }
}

/// Exposto para servidor.rs injetar _CORPO_JSON
pub fn json_deserializar_val(s: &str) -> Result<Valor, String> {
    json_para_valor(s)
}

fn json_para_valor(s: &str) -> Result<Valor, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("JSON vazio".to_string());
    }

    match s.as_bytes().first() {
        Some(b'n') if s == "null" => Ok(Valor::Nulo),
        Some(b't') if s == "true" => Ok(Valor::Booleano(true)),
        Some(b'f') if s == "false" => Ok(Valor::Booleano(false)),
        Some(b'"') => Ok(Valor::Texto(json_ler_string(s)?)),
        Some(b'[') => json_ler_array(s),
        Some(b'{') => json_ler_objeto(s),
        _ if !s.contains(['.', 'e', 'E']) => s
            .parse::<i64>()
            .map(Valor::Inteiro)
            .map_err(|_| format!("JSON invalido: {}", &s[..s.len().min(40)])),
        _ => s
            .parse::<f64>()
            .map(Valor::Numero)
            .map_err(|_| format!("JSON invalido: {}", &s[..s.len().min(40)])),
    }
}

fn json_ler_string(s: &str) -> Result<String, String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.first() != Some(&'"') {
        return Err("String JSON deve comecar com \"".to_string());
    }
    let mut resultado = String::new();
    let mut i = 1;
    while i < chars.len() {
        match chars[i] {
            '"' => return Ok(resultado),
            '\\' => {
                i += 1;
                match chars.get(i) {
                    Some('"') => resultado.push('"'),
                    Some('\\') => resultado.push('\\'),
                    Some('/') => resultado.push('/'),
                    Some('n') => resultado.push('\n'),
                    Some('r') => resultado.push('\r'),
                    Some('t') => resultado.push('\t'),
                    Some('u') => {
                        let hex: String = chars
                            .get(i + 1..i + 5)
                            .map(|c| c.iter().collect())
                            .unwrap_or_default();
                        if let Ok(n) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(n) {
                                resultado.push(c);
                            }
                        }
                        i += 4;
                    }
                    Some(c) => {
                        resultado.push('\\');
                        resultado.push(*c);
                    }
                    None => break,
                }
            }
            c => resultado.push(c),
        }
        i += 1;
    }
    Ok(resultado)
}

fn json_ler_array(s: &str) -> Result<Valor, String> {
    let mut lista = Vec::new();
    let s = s.trim();
    if s == "[]" {
        return Ok(Valor::Lista(lista));
    }
    // Simples: encontrar tokens de nivel 0
    let interior = &s[1..s.len().saturating_sub(1)];
    for item in json_split_nivel0(interior, ',') {
        let item = item.trim();
        if !item.is_empty() {
            lista.push(json_para_valor(item)?);
        }
    }
    Ok(Valor::Lista(lista))
}

fn json_ler_objeto(s: &str) -> Result<Valor, String> {
    let mut mapa = HashMap::new();
    let s = s.trim();
    if s == "{}" {
        return Ok(Valor::Mapa(mapa));
    }
    let interior = &s[1..s.len().saturating_sub(1)];
    for par in json_split_nivel0(interior, ',') {
        let par = par.trim();
        if par.is_empty() {
            continue;
        }
        if let Some(pos) = json_posicao_dois_pontos(par) {
            let chave_str = par[..pos].trim();
            let val_str = par[pos + 1..].trim();
            let chave = json_ler_string(chave_str)?;
            let val = json_para_valor(val_str)?;
            mapa.insert(chave, val);
        }
    }
    Ok(Valor::Mapa(mapa))
}

fn json_split_nivel0(s: &str, sep: char) -> Vec<&str> {
    let mut partes = Vec::new();
    let mut depth = 0i32;
    let mut em_string = false;
    let mut escape = false;
    let mut inicio = 0;
    for (i, c) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if em_string {
            if c == '\\' {
                escape = true;
            } else if c == '"' {
                em_string = false;
            }
            continue;
        }
        match c {
            '"' => em_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            c if c == sep && depth == 0 => {
                partes.push(&s[inicio..i]);
                inicio = i + c.len_utf8();
            }
            _ => {}
        }
    }
    partes.push(&s[inicio..]);
    partes
}

fn json_posicao_dois_pontos(s: &str) -> Option<usize> {
    let mut em_string = false;
    let mut escape = false;
    for (i, c) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if em_string {
            if c == '\\' {
                escape = true;
            } else if c == '"' {
                em_string = false;
            }
            continue;
        }
        if c == '"' {
            em_string = true;
        } else if c == ':' {
            return Some(i);
        }
    }
    None
}

// -- URL decode simples --------------------------------------------------------

fn url_decodificar_simples(s: &str) -> String {
    let mut bytes: Vec<u8> = Vec::new();
    let mut iter = s.bytes();
    while let Some(b) = iter.next() {
        match b {
            b'%' => {
                let h1 = iter.next().unwrap_or(b'0') as char;
                let h2 = iter.next().unwrap_or(b'0') as char;
                if let Ok(byte) = u8::from_str_radix(&format!("{}{}", h1, h2), 16) {
                    bytes.push(byte);
                }
            }
            b'+' => bytes.push(b' '),
            c => bytes.push(c),
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erro_runtime_inclui_linha_e_contexto() {
        let fonte = "var a = 1\nimprimir(inexistente)";
        let mut lexer = crate::lexer::Lexer::novo(fonte);
        let tokens = lexer.tokenizar().unwrap();
        let mut parser = crate::parser::Parser::novo(tokens);
        let programa = parser.parsear().unwrap();
        let erro = Interpretador::novo().executar(&programa).unwrap_err();
        assert!(erro.contains("Erro na linha 2"));
        assert!(erro.contains("imprimir(inexistente)"));
    }
}

// -- Helpers de tensor --------------------------------------------------------

/// Renderizacao recursiva de tensor n-dimensional para Display.
fn tensor_fmt(shape: &[usize], dados: &[f64], dim: usize, offset: &mut usize) -> String {
    if dim == shape.len() - 1 {
        let mut s = String::from("[");
        for i in 0..shape[dim] {
            if i > 0 {
                s.push_str(", ");
            }
            let v = dados[*offset];
            *offset += 1;
            if v.fract() == 0.0 && v.abs() < 1e15 {
                s.push_str(&format!("{}", v as i64));
            } else {
                s.push_str(&format!("{}", v));
            }
        }
        s.push(']');
        s
    } else {
        let mut s = String::from("[");
        for i in 0..shape[dim] {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&tensor_fmt(shape, dados, dim + 1, offset));
        }
        s.push(']');
        s
    }
}

fn tensor_transpor_2d(linhas: usize, colunas: usize, dados: &[f64]) -> Valor {
    let mut nd = vec![0.0f64; linhas * colunas];
    for i in 0..linhas {
        for j in 0..colunas {
            nd[j * linhas + i] = dados[i * colunas + j];
        }
    }
    Valor::Tensor {
        shape: vec![colunas, linhas],
        dados: Arc::new(nd),
    }
}

/// Multiplicacao de matrizes 2D usando ndarray (SIMD auto-vectorized).
fn tensor_matmul_ndarray(
    la: usize,
    ca: usize,
    da: &[f64],
    _lb: usize,
    cb: usize,
    db: &[f64],
) -> Valor {
    use ndarray::ArrayView2;
    let av = ArrayView2::from_shape((la, ca), da).expect("shape invalido A");
    let bv = ArrayView2::from_shape((ca, cb), db).expect("shape invalido B");
    let c = av.dot(&bv);
    Valor::Tensor {
        shape: vec![la, cb],
        dados: Arc::new(c.into_raw_vec()),
    }
}

fn tensor_para_lista_val(shape: &[usize], dados: &[f64]) -> Valor {
    if shape.len() == 1 {
        Valor::Lista((0..shape[0]).map(|i| Valor::Numero(dados[i])).collect())
    } else {
        let stride: usize = shape[1..].iter().product();
        Valor::Lista(
            (0..shape[0])
                .map(|i| tensor_para_lista_val(&shape[1..], &dados[i * stride..(i + 1) * stride]))
                .collect(),
        )
    }
}

/// Extrai shape de args: aceita ([dim1, dim2]) OU (dim1, dim2, ...).
fn shape_de_args(args: &[Valor], nome: &str) -> Result<Vec<usize>, String> {
    if args.len() == 1 {
        match &args[0] {
            Valor::Lista(l) => l.iter().map(|v| to_usize(v, nome)).collect(),
            v => Ok(vec![to_usize(v, nome)?]),
        }
    } else {
        args.iter().map(|v| to_usize(v, nome)).collect()
    }
}

/// Reducao ao longo de um eixo. Para tensores 2D: eixo 0 = por colunas, eixo 1 = por linhas.
fn tensor_reducao_eixo(
    shape: &[usize],
    dados: &[f64],
    eixo: usize,
    f: impl Fn(f64, f64) -> f64,
    init: f64,
) -> Result<Valor, String> {
    if eixo >= shape.len() {
        return Err(format!(
            "eixo {} invalido para tensor {}D",
            eixo,
            shape.len()
        ));
    }
    if shape.len() != 2 {
        return Err("reducao_eixo: apenas tensores 2D suportados por ora".to_string());
    }
    let (nlin, ncol) = (shape[0], shape[1]);
    match eixo {
        0 => {
            // Soma ao longo das linhas → resultado com shape [ncol]
            let mut r = vec![init; ncol];
            for i in 0..nlin {
                for j in 0..ncol {
                    r[j] = f(r[j], dados[i * ncol + j]);
                }
            }
            Ok(Valor::Tensor {
                shape: vec![ncol],
                dados: Arc::new(r),
            })
        }
        1 => {
            // Soma ao longo das colunas → resultado com shape [nlin]
            let mut r = vec![init; nlin];
            for i in 0..nlin {
                for j in 0..ncol {
                    r[i] = f(r[i], dados[i * ncol + j]);
                }
            }
            Ok(Valor::Tensor {
                shape: vec![nlin],
                dados: Arc::new(r),
            })
        }
        _ => unreachable!(),
    }
}

fn tensor_concat(tensores: &[(Vec<usize>, Arc<Vec<f64>>)], eixo: usize) -> Result<Valor, String> {
    if tensores.is_empty() {
        return Err("tensor_concatenar: lista vazia".to_string());
    }
    let shape0 = &tensores[0].0;
    if eixo >= shape0.len() {
        return Err(format!("tensor_concatenar: eixo {} invalido", eixo));
    }
    // Valida compatibilidade de shape em todas as dims exceto eixo
    for (s, _) in &tensores[1..] {
        if s.len() != shape0.len() {
            return Err("tensor_concatenar: tensores com numero de dims diferentes".to_string());
        }
        for (i, (&a, &b)) in shape0.iter().zip(s.iter()).enumerate() {
            if i != eixo && a != b {
                return Err(format!(
                    "tensor_concatenar: dim {} incompativel ({} vs {})",
                    i, a, b
                ));
            }
        }
    }
    let mut nova_shape = shape0.clone();
    nova_shape[eixo] = tensores.iter().map(|(s, _)| s[eixo]).sum();
    let mut dados: Vec<f64> = Vec::with_capacity(nova_shape.iter().product());

    // Para tensores 2D (caso mais comum), concatena por eixo
    if shape0.len() == 2 {
        match eixo {
            0 => {
                for (_, d) in tensores {
                    dados.extend_from_slice(d);
                }
            }
            1 => {
                let nlin = shape0[0];
                for i in 0..nlin {
                    for (s, d) in tensores {
                        let nc = s[1];
                        dados.extend_from_slice(&d[i * nc..(i + 1) * nc]);
                    }
                }
            }
            _ => return Err("tensor_concatenar: eixo invalido para 2D".to_string()),
        }
    } else {
        return Err("tensor_concatenar: apenas tensores 2D suportados por ora".to_string());
    }
    Ok(Valor::Tensor {
        shape: nova_shape,
        dados: Arc::new(dados),
    })
}

// -- Helpers para funcoes nativas ---------------------------------------------

fn lista_f64(v: &Valor, contexto: &str) -> Result<Vec<f64>, String> {
    match v {
        Valor::Lista(l) => l.iter().map(|x| to_f64(x, contexto)).collect(),
        _ => Err(format!(
            "'{}' requer lista de numeros, recebeu {}",
            contexto, v
        )),
    }
}

fn to_f64(v: &Valor, contexto: &str) -> Result<f64, String> {
    match v {
        Valor::Numero(n) => Ok(*n),
        Valor::Inteiro(n) => Ok(*n as f64),
        _ => Err(format!("'{}' requer numero, recebeu {}", contexto, v)),
    }
}

fn to_usize(v: &Valor, contexto: &str) -> Result<usize, String> {
    match v {
        Valor::Inteiro(n) if *n >= 0 => Ok(*n as usize),
        Valor::Numero(n) if *n >= 0.0 => Ok(*n as usize),
        _ => Err(format!(
            "'{}' requer inteiro nao-negativo, recebeu {}",
            contexto, v
        )),
    }
}

/// Formata uma string com substituicoes de chaves: `{chave}`, `{chave:>N}`, `{chave:<N}`, `{chave:.Nf}`.
fn formatar_padrao(padrao: &str, mapa: &HashMap<String, Valor>) -> Result<String, String> {
    let mut resultado = String::with_capacity(padrao.len() + 32);
    let mut chars = padrao.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            if chars.peek() == Some(&'{') {
                chars.next();
                resultado.push('{');
                continue;
            }
            let mut spec = String::new();
            for c2 in chars.by_ref() {
                if c2 == '}' {
                    break;
                }
                spec.push(c2);
            }
            // spec pode ser "chave", "chave:>N", "chave:<N", "chave:.Nf"
            let (chave, fmt_spec) = match spec.split_once(':') {
                Some((k, s)) => (k.trim(), Some(s.trim())),
                None => (spec.trim(), None),
            };
            let valor = mapa.get(chave).cloned().unwrap_or(Valor::Nulo);
            let texto = valor.to_string();
            let formatado = match fmt_spec {
                None => texto,
                Some(spec) => {
                    if let Some(rest) = spec.strip_prefix('>') {
                        let n: usize = rest.parse().unwrap_or(0);
                        format!("{:>width$}", texto, width = n)
                    } else if let Some(rest) = spec.strip_prefix('<') {
                        let n: usize = rest.parse().unwrap_or(0);
                        format!("{:<width$}", texto, width = n)
                    } else if let Some(rest) = spec.strip_prefix('.') {
                        let rest = rest.trim_end_matches('f');
                        let n: usize = rest.parse().unwrap_or(2);
                        if let Ok(f) = texto.parse::<f64>() {
                            format!("{:.prec$}", f, prec = n)
                        } else {
                            texto
                        }
                    } else {
                        texto
                    }
                }
            };
            resultado.push_str(&formatado);
        } else if c == '}' && chars.peek() == Some(&'}') {
            chars.next();
            resultado.push('}');
        } else {
            resultado.push(c);
        }
    }
    Ok(resultado)
}

/// MD5 puro sem dependencia externa (128 bits → 16 bytes).
fn md5_simples(data: &[u8]) -> u128 {
    // Implementacao RFC 1321 compacta
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());
    let (mut a0, mut b0, mut c0, mut d0) =
        (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    for chunk in msg.chunks(64) {
        let mut m = [0u32; 16];
        for (i, w) in m.iter_mut().enumerate() {
            *w = u32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0u32..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | (!b & d), i),
                16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let temp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                (a.wrapping_add(f)
                    .wrapping_add(K[i as usize])
                    .wrapping_add(m[g as usize]))
                .rotate_left(S[i as usize]),
            );
            a = temp;
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let r = [
        a0.to_le_bytes(),
        b0.to_le_bytes(),
        c0.to_le_bytes(),
        d0.to_le_bytes(),
    ];
    let bytes: Vec<u8> = r.iter().flatten().copied().collect();
    u128::from_le_bytes(bytes.try_into().unwrap())
}
