/// Servidor HTTP embutido da linguagem PEP — thread pool + execucao in-process
///
/// Uso: pep servir [porta] [diretorio]
///
/// Variaveis PEP injetadas em cada requisicao:
///   _GET      -> mapa com query string
///   _POST     -> mapa com corpo application/x-www-form-urlencoded
///   _URL      -> caminho da requisicao ("/pagina.phtml")
///   _METODO   -> "GET", "POST", etc.
///   _COOKIE   -> mapa de cookies
///   _SESSAO_ID -> ID de sessao atual
///   _SERVIDOR -> verdadeiro
use crate::interpretador::{drenar_resposta, iniciar_contexto_servidor};
use crate::vm::{valor_para_vm, Maquina, VmValor};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant, SystemTime};

const LIMITE_CORPO_PADRAO: usize = 4 * 1024 * 1024;
const LIMITE_CABECALHOS: usize = 64 * 1024;

static CACHE_BYTECODE: OnceLock<RwLock<HashMap<PathBuf, (SystemTime, Vec<crate::bytecode::Op>)>>> =
    OnceLock::new();

// ── Mapa global de aplicação (thread-safe, TTL opcional) ──────────────────────
// Acessado por global_definir / global_obter / global_apagar / global_listar
struct EntradaGlobal {
    valor: VmValor,
    expira: Option<Instant>,
}

static GLOBAIS_APP: OnceLock<RwLock<HashMap<String, EntradaGlobal>>> = OnceLock::new();
fn globais_app() -> &'static RwLock<HashMap<String, EntradaGlobal>> {
    GLOBAIS_APP.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn global_definir(chave: String, valor: VmValor, ttl_seg: Option<u64>) {
    let expira = ttl_seg.map(|s| Instant::now() + Duration::from_secs(s));
    globais_app()
        .write()
        .unwrap()
        .insert(chave, EntradaGlobal { valor, expira });
}
pub fn global_obter(chave: &str) -> Option<VmValor> {
    let mapa = globais_app().read().unwrap();
    mapa.get(chave).and_then(|e| {
        if e.expira.map(|t| Instant::now() > t).unwrap_or(false) {
            None
        } else {
            Some(e.valor.clone())
        }
    })
}
pub fn global_apagar(chave: &str) {
    globais_app().write().unwrap().remove(chave);
}
pub fn global_existe(chave: &str) -> bool {
    global_obter(chave).is_some()
}
pub fn global_listar() -> Vec<String> {
    globais_app().read().unwrap().keys().cloned().collect()
}

fn cache_bytecode() -> &'static RwLock<HashMap<PathBuf, (SystemTime, Vec<crate::bytecode::Op>)>> {
    CACHE_BYTECODE.get_or_init(|| RwLock::new(HashMap::new()))
}

static WS_ATIVOS: AtomicUsize = AtomicUsize::new(0);

struct GuardaWs;
impl Drop for GuardaWs {
    fn drop(&mut self) {
        WS_ATIVOS.fetch_sub(1, Ordering::AcqRel);
    }
}

fn reservar_websocket() -> Option<GuardaWs> {
    let maximo = std::env::var("PEP_WS_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    WS_ATIVOS
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |atual| {
            (atual < maximo).then_some(atual + 1)
        })
        .ok()
        .map(|_| GuardaWs)
}

struct HandlerEnviavel(VmValor);
// VmValor contem somente dados owned (Arc) sem interior mutability.
unsafe impl Send for HandlerEnviavel {}

// -- Registry global de rotas --------------------------------------------------

struct Rota {
    metodo: String,
    padrao: String,
    handler: VmValor,
}

// SAFETY: VmValor e Send+Sync (todos os seus campos sao owned sem interior mutability).
unsafe impl Send for Rota {}
unsafe impl Sync for Rota {}

static ROTAS: OnceLock<Arc<RwLock<Vec<Rota>>>> = OnceLock::new();

fn rotas_arc() -> Arc<RwLock<Vec<Rota>>> {
    ROTAS
        .get_or_init(|| Arc::new(RwLock::new(Vec::new())))
        .clone()
}

// ── Registry de middlewares (usar) ────────────────────────────────────────────

struct MwEnviavel(VmValor);
unsafe impl Send for MwEnviavel {}
unsafe impl Sync for MwEnviavel {}

static MIDDLEWARES: OnceLock<RwLock<Vec<VmValor>>> = OnceLock::new();

fn middlewares() -> &'static RwLock<Vec<VmValor>> {
    MIDDLEWARES.get_or_init(|| RwLock::new(Vec::new()))
}

pub fn registrar_middleware(handler: VmValor) {
    middlewares().write().unwrap().push(handler);
}

// Flag thread-local: o middleware atual chamou proximo()?
use std::cell::{Cell, RefCell};
thread_local! {
    static PROXIMO_CHAMADO: Cell<bool> = const { Cell::new(false) };
}

pub fn sinalizar_proximo() {
    PROXIMO_CHAMADO.with(|p| p.set(true));
}
fn limpar_proximo() {
    PROXIMO_CHAMADO.with(|p| p.set(false));
}
fn proximo_foi_chamado() -> bool {
    PROXIMO_CHAMADO.with(|p| p.get())
}

// ── Stream SSE thread-local ────────────────────────────────────────────────────
// Cada worker thread pode estar atendendo uma conexão SSE; o stream fica aqui
// enquanto o handler PEP itera e chama sse_enviar().

thread_local! {
    static SSE_STREAM: RefCell<Option<TcpStream>> = RefCell::new(None);
}

fn sse_definir_stream(s: TcpStream) {
    SSE_STREAM.with(|ss| *ss.borrow_mut() = Some(s));
}
fn sse_limpar_stream() {
    SSE_STREAM.with(|ss| *ss.borrow_mut() = None);
}

/// Envia os cabeçalhos SSE e retorna true se o stream foi configurado.
pub fn sse_iniciar() -> bool {
    SSE_STREAM.with(|ss| {
        if let Some(stream) = ss.borrow_mut().as_mut() {
            let hdrs = "HTTP/1.1 200 OK\r\n\
                        Content-Type: text/event-stream; charset=utf-8\r\n\
                        Cache-Control: no-cache\r\n\
                        Connection: keep-alive\r\n\
                        X-Accel-Buffering: no\r\n\
                        \r\n";
            stream.write_all(hdrs.as_bytes()).is_ok()
        } else {
            false
        }
    })
}

/// Serializa `valor` como evento SSE `data: <json>\n\n` e faz flush.
pub fn sse_enviar(valor: &VmValor) -> bool {
    let dados = match valor {
        VmValor::Str(s) => s.clone(),
        v => {
            // Serializa como JSON simples (reutiliza Display que já usa repr interno)
            v.to_string()
        }
    };
    let linha = format!("data: {}\n\n", dados);
    SSE_STREAM.with(|ss| {
        if let Some(stream) = ss.borrow_mut().as_mut() {
            stream.write_all(linha.as_bytes()).is_ok() && stream.flush().is_ok()
        } else {
            false
        }
    })
}

/// Envia evento SSE com campo `event:` customizado.
pub fn sse_enviar_evento(evento: &str, valor: &VmValor) -> bool {
    let dados = match valor {
        VmValor::Str(s) => s.clone(),
        v => v.to_string(),
    };
    let linha = format!("event: {}\ndata: {}\n\n", evento, dados);
    SSE_STREAM.with(|ss| {
        if let Some(stream) = ss.borrow_mut().as_mut() {
            stream.write_all(linha.as_bytes()).is_ok() && stream.flush().is_ok()
        } else {
            false
        }
    })
}

/// Fecha a conexão SSE (limpa o stream).
pub fn sse_fechar() {
    sse_limpar_stream();
}

// -- Registry de páginas estilo PHP (pages/*.pep) ------------------------------

static PAGINAS: OnceLock<Arc<RwLock<Vec<(String, PathBuf)>>>> = OnceLock::new();
static RAIZ_DOCUMENTO: OnceLock<RwLock<Option<PathBuf>>> = OnceLock::new();

fn paginas_arc() -> Arc<RwLock<Vec<(String, PathBuf)>>> {
    PAGINAS
        .get_or_init(|| Arc::new(RwLock::new(Vec::new())))
        .clone()
}

fn raiz_documento() -> &'static RwLock<Option<PathBuf>> {
    RAIZ_DOCUMENTO.get_or_init(|| RwLock::new(None))
}

fn encontrar_pagina(caminho: &str) -> Option<(PathBuf, HashMap<String, String>)> {
    let raiz = raiz_documento().read().unwrap().clone();
    if let Some(raiz) = raiz.as_ref() {
        if let Some(script) = resolver_script_na_raiz(raiz, caminho) {
            return Some(script);
        }
    }
    let arc = paginas_arc();
    let lista = arc.read().unwrap();
    for (padrao, arquivo) in lista.iter() {
        if let Some(params) = combinar_padrao(padrao, caminho) {
            return Some((arquivo.clone(), params));
        }
    }
    None
}

fn resolver_script_na_raiz(
    raiz: &Path,
    caminho: &str,
) -> Option<(PathBuf, HashMap<String, String>)> {
    let raiz = raiz.canonicalize().ok()?;
    let url = url_decodificar(caminho);
    let segmentos: Vec<&str> = url
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    if segmentos
        .iter()
        .any(|s| *s == "." || *s == ".." || s.contains(['\\', '\0']))
    {
        return None;
    }
    let mut diretorio = raiz.clone();
    let mut params = HashMap::new();

    if segmentos.is_empty() {
        return primeiro_script_valido(&raiz, &raiz, &["index.pep", "index.phtml"])
            .map(|p| (p, params));
    }

    for segmento in &segmentos[..segmentos.len() - 1] {
        let exato = diretorio.join(segmento);
        if exato.is_dir() {
            diretorio = exato;
            continue;
        }
        let (dinamico, nome) = primeiro_diretorio_dinamico(&diretorio)?;
        params.insert(nome, url_decodificar(segmento));
        diretorio = dinamico;
    }

    let ultimo = segmentos[segmentos.len() - 1];
    let mut candidatos = Vec::new();
    if ultimo.ends_with(".pep") || ultimo.ends_with(".phtml") {
        candidatos.push(diretorio.join(ultimo));
    } else {
        candidatos.push(diretorio.join(format!("{}.pep", ultimo)));
        candidatos.push(diretorio.join(format!("{}.phtml", ultimo)));
        candidatos.push(diretorio.join(ultimo).join("index.pep"));
        candidatos.push(diretorio.join(ultimo).join("index.phtml"));
    }
    for candidato in candidatos {
        if let Some(real) = script_dentro_da_raiz(&raiz, &candidato) {
            return Some((real, params));
        }
    }

    let mut dinamicos: Vec<_> = std::fs::read_dir(&diretorio)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            let nome = path.file_stem()?.to_str()?.to_string();
            let ext = path.extension()?.to_str()?;
            (path.is_file()
                && matches!(ext, "pep" | "phtml")
                && nome.starts_with('[')
                && nome.ends_with(']'))
            .then(|| (path, nome[1..nome.len() - 1].to_string()))
        })
        .collect();
    dinamicos.sort_by(|a, b| a.0.cmp(&b.0));
    if let Some((arquivo, nome)) = dinamicos.into_iter().next() {
        params.insert(nome, url_decodificar(ultimo));
        return script_dentro_da_raiz(&raiz, &arquivo).map(|p| (p, params));
    }

    let dir_exato = diretorio.join(ultimo);
    let (dir_dinamico, nome) = if dir_exato.is_dir() {
        (dir_exato, None)
    } else {
        let (dir, nome) = primeiro_diretorio_dinamico(&diretorio)?;
        (dir, Some(nome))
    };
    if let Some(nome) = nome {
        params.insert(nome, url_decodificar(ultimo));
    }
    primeiro_script_valido(&raiz, &dir_dinamico, &["index.pep", "index.phtml"]).map(|p| (p, params))
}

fn primeiro_diretorio_dinamico(diretorio: &Path) -> Option<(PathBuf, String)> {
    let mut dirs: Vec<_> = std::fs::read_dir(diretorio)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            let nome = e.file_name().to_string_lossy().to_string();
            (path.is_dir() && nome.starts_with('[') && nome.ends_with(']'))
                .then(|| (path, nome[1..nome.len() - 1].to_string()))
        })
        .collect();
    dirs.sort_by(|a, b| a.0.cmp(&b.0));
    dirs.into_iter().next()
}

fn primeiro_script_valido(raiz: &Path, diretorio: &Path, nomes: &[&str]) -> Option<PathBuf> {
    nomes
        .iter()
        .find_map(|nome| script_dentro_da_raiz(raiz, &diretorio.join(nome)))
}

fn script_dentro_da_raiz(raiz: &Path, candidato: &Path) -> Option<PathBuf> {
    let real = candidato.canonicalize().ok()?;
    let ext = real.extension().and_then(|e| e.to_str())?;
    (real.starts_with(raiz) && real.is_file() && matches!(ext, "pep" | "phtml")).then_some(real)
}

/// Converte caminho de arquivo em padrão de rota.
/// pages/index.pep          → /
/// pages/sobre.pep          → /sobre
/// pages/api/ping.pep       → /api/ping
/// pages/api/tarefas/index.pep → /api/tarefas
/// pages/usuario/[id].pep   → /usuario/:id
fn arquivo_para_padrao(raiz: &Path, arquivo: &Path) -> String {
    let rel = arquivo.strip_prefix(raiz).unwrap_or(arquivo);
    let mut partes: Vec<String> = rel
        .with_extension("")
        .components()
        .map(|c| {
            let s = c.as_os_str().to_string_lossy().to_string();
            if s.starts_with('[') && s.ends_with(']') {
                format!(":{}", &s[1..s.len() - 1])
            } else {
                s
            }
        })
        .collect();
    if partes.last().map(|s| s == "index").unwrap_or(false) {
        partes.pop();
    }
    if partes.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", partes.join("/"))
    }
}

fn descobrir_paginas(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut resultado = Vec::new();
    descobrir_recursivo(dir, dir, &mut resultado);
    // ordena: rotas fixas antes das dinâmicas (ex: /api/ping antes de /api/:id)
    resultado.sort_by(|(a, _), (b, _)| {
        let dinamicos_a = a.matches(':').count();
        let dinamicos_b = b.matches(':').count();
        dinamicos_a.cmp(&dinamicos_b).then(a.cmp(b))
    });
    resultado
}

fn descobrir_recursivo(raiz: &Path, atual: &Path, out: &mut Vec<(String, PathBuf)>) {
    let entries = match std::fs::read_dir(atual) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut entradas: Vec<_> = entries.flatten().collect();
    entradas.sort_by_key(|e| e.file_name());
    for entry in entradas {
        let path = entry.path();
        if path.is_dir() {
            descobrir_recursivo(raiz, &path, out);
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("pep" | "phtml")
        ) {
            let padrao = arquivo_para_padrao(raiz, &path);
            out.push((padrao, path));
        }
    }
}

pub fn carregar_paginas_da_raiz(raiz: &Path) {
    *raiz_documento().write().unwrap() =
        Some(raiz.canonicalize().unwrap_or_else(|_| raiz.to_path_buf()));
    let paginas = descobrir_paginas(&raiz.join("pages"));
    let arc = paginas_arc();
    let mut lista = arc.write().unwrap();
    lista.clear();
    if !paginas.is_empty() {
        println!("Paginas descobertas em pages/:");
    }
    for (padrao, arquivo) in paginas {
        println!("  [pagina] {}", padrao);
        lista.push((padrao, arquivo));
    }
}

// -- Registry global de modelos de IA -----------------------------------------
// Modelos carregados uma vez (na inicializacao do servidor) e compartilhados
// entre todas as threads de requisicao sem copia. Clone de Valor::Tensor = O(1).
// Armazenados como crate::interpretador::Valor para compatibilidade com as funções
// nativas do interpretador (modelo_carregar etc.) acessadas pelo bridge da VM.

static MODELOS: OnceLock<Arc<RwLock<HashMap<String, crate::interpretador::Valor>>>> =
    OnceLock::new();

fn modelos_arc() -> Arc<RwLock<HashMap<String, crate::interpretador::Valor>>> {
    MODELOS
        .get_or_init(|| Arc::new(RwLock::new(HashMap::new())))
        .clone()
}

pub fn modelo_definir(nome: String, valor: crate::interpretador::Valor) {
    modelos_arc().write().unwrap().insert(nome, valor);
}

pub fn modelo_obter(nome: &str) -> Option<crate::interpretador::Valor> {
    modelos_arc().read().unwrap().get(nome).cloned()
}

pub fn modelo_remover(nome: &str) {
    modelos_arc().write().unwrap().remove(nome);
}

pub fn modelos_listar() -> Vec<String> {
    modelos_arc().read().unwrap().keys().cloned().collect()
}

pub fn registrar_rota(metodo: String, padrao: String, handler: VmValor) {
    println!("  [rota] {} {}", metodo, padrao);
    rotas_arc().write().unwrap().push(Rota {
        metodo,
        padrao,
        handler,
    });
}

pub fn carregar_rotas_da_raiz(raiz: &Path) {
    let routes_file = raiz.join("routes.pep");
    if !routes_file.exists() {
        return;
    }
    println!("Carregando rotas de routes.pep...");
    rotas_arc().write().unwrap().clear();
    match std::fs::read_to_string(&routes_file) {
        Ok(fonte) => match compilar_bytecode_fonte(&fonte, raiz) {
            Ok(ops) => {
                let mut vm = Maquina::com_base(raiz.to_path_buf());
                if let Err(e) = vm.executar(&ops) {
                    eprintln!("Erro ao carregar routes.pep: {}", e);
                }
            }
            Err(e) => eprintln!("Erro de sintaxe em routes.pep: {}", e),
        },
        Err(e) => eprintln!("Nao foi possivel ler routes.pep: {}", e),
    }
}

fn encontrar_rota(metodo: &str, caminho: &str) -> Option<(VmValor, HashMap<String, String>)> {
    let rotas = rotas_arc();
    let lista = rotas.read().unwrap();
    for r in lista.iter() {
        if r.metodo != "*" && !r.metodo.eq_ignore_ascii_case(metodo) {
            continue;
        }
        if let Some(params) = combinar_padrao(&r.padrao, caminho) {
            return Some((r.handler.clone(), params));
        }
    }
    None
}

fn metodos_rota(caminho: &str) -> Vec<String> {
    let rotas = rotas_arc();
    let lista = rotas.read().unwrap();
    let mut metodos: Vec<String> = lista
        .iter()
        .filter(|r| combinar_padrao(&r.padrao, caminho).is_some())
        .map(|r| r.metodo.clone())
        .collect();
    if metodos.iter().any(|m| m == "GET") && !metodos.iter().any(|m| m == "HEAD") {
        metodos.push("HEAD".to_string());
    }
    if !metodos.is_empty() {
        metodos.push("OPTIONS".to_string());
    }
    metodos.sort();
    metodos.dedup();
    metodos
}

fn combinar_padrao(padrao: &str, caminho: &str) -> Option<HashMap<String, String>> {
    // Wildcard: "/prefixo/*"
    if let Some(prefixo) = padrao.strip_suffix("/*") {
        if caminho.starts_with(prefixo) {
            return Some(HashMap::new());
        }
        return None;
    }

    let partes_p: Vec<&str> = padrao
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let partes_c: Vec<&str> = caminho
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();

    if partes_p.len() != partes_c.len() {
        return None;
    }

    let mut params = HashMap::new();
    for (p, c) in partes_p.iter().zip(partes_c.iter()) {
        if let Some(nome) = p.strip_prefix(':') {
            params.insert(nome.to_string(), url_decodificar(c));
        } else if *p != *c {
            return None;
        }
    }
    Some(params)
}

// -- Ponto de entrada ----------------------------------------------------------

pub fn iniciar(porta: u16, raiz: PathBuf) {
    let n_workers: usize = std::env::var("PEP_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .max(2)
        });

    let raiz = Arc::new(raiz.canonicalize().unwrap_or(raiz));

    // Inicializa o repositorio de sessoes in-memory (TTL default 30 min ou PEP_SESSAO_TTL)
    let ttl: u64 = std::env::var("PEP_SESSAO_TTL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    crate::sessoes::inicializar(ttl);

    // Rotas explicitas precedem o fallback baseado em pages/.
    carregar_rotas_da_raiz(raiz.as_ref());

    // Descobre páginas estilo PHP em pages/ (arquivo = rota, sem routes.pep)
    carregar_paginas_da_raiz(raiz.as_ref());

    let endereco = format!("0.0.0.0:{}", porta);
    let listener = match TcpListener::bind(&endereco) {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "Nao foi possivel iniciar o servidor na porta {}: {}",
                porta, e
            );
            std::process::exit(1);
        }
    };

    // Canal limitado: faz backpressure automatico quando todos os workers estao ocupados.
    // Capacidade = 4x workers para absorver picos sem consumir memoria indefinidamente.
    let (tx, rx) = mpsc::sync_channel::<TcpStream>(n_workers * 4);
    let rx = Arc::new(Mutex::new(rx));

    println!("Servidor PEP rodando em http://localhost:{}", porta);
    println!("Servindo arquivos de: {}", raiz.display());
    println!(
        "{} workers | fila max {} | Ctrl+C para parar\n",
        n_workers,
        n_workers * 4
    );

    for _ in 0..n_workers {
        let rx = Arc::clone(&rx);
        let raiz = Arc::clone(&raiz);
        std::thread::spawn(move || loop {
            let stream = match rx.lock().unwrap().recv() {
                Ok(s) => s,
                Err(_) => break,
            };
            tratar(stream, &raiz);
        });
    }

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                // send() bloqueia se a fila estiver cheia, aplicando backpressure.
                // Conexoes sao aceitas pelo kernel enquanto bloqueado.
                if tx.send(s).is_err() {
                    break;
                }
            }
            Err(e) => eprintln!("Erro de conexao: {}", e),
        }
    }
}

// -- Requisicao HTTP parseada --------------------------------------------------

struct Requisicao {
    metodo: String,
    caminho_url: String,
    query_str: String,
    cabecalhos: HashMap<String, String>,
    corpo: Vec<u8>,
}

enum ErroRequisicao {
    Invalida,
    CabecalhosGrandes,
    CorpoGrande,
    CodificacaoNaoSuportada,
}

fn ler_requisicao(stream: &TcpStream) -> Result<Requisicao, ErroRequisicao> {
    let mut reader = BufReader::new(stream);

    let mut primeira = String::new();
    reader
        .read_line(&mut primeira)
        .map_err(|_| ErroRequisicao::Invalida)?;
    if primeira.len() > 8 * 1024 {
        return Err(ErroRequisicao::CabecalhosGrandes);
    }
    let partes: Vec<&str> = primeira.trim().split_whitespace().collect();
    if partes.len() != 3 || !partes[2].starts_with("HTTP/1.") {
        return Err(ErroRequisicao::Invalida);
    }

    let metodo = partes[0].to_string();
    let url_completa = partes[1];
    let (caminho_url, query_str) = match url_completa.split_once('?') {
        Some((c, q)) => (c.to_string(), q.to_string()),
        None => (url_completa.to_string(), String::new()),
    };

    let mut cabecalhos: HashMap<String, String> = HashMap::new();
    let mut total_cabecalhos = primeira.len();
    loop {
        let mut linha = String::new();
        reader
            .read_line(&mut linha)
            .map_err(|_| ErroRequisicao::Invalida)?;
        total_cabecalhos = total_cabecalhos.saturating_add(linha.len());
        if total_cabecalhos > LIMITE_CABECALHOS || linha.len() > 8 * 1024 {
            return Err(ErroRequisicao::CabecalhosGrandes);
        }
        let linha = linha.trim();
        if linha.is_empty() {
            break;
        }
        if let Some((k, v)) = linha.split_once(':') {
            cabecalhos.insert(k.trim().to_lowercase(), v.trim().to_string());
        }
    }

    if cabecalhos
        .get("transfer-encoding")
        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
    {
        return Err(ErroRequisicao::CodificacaoNaoSuportada);
    }

    let limite_corpo = std::env::var("PEP_MAX_BODY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(LIMITE_CORPO_PADRAO);
    let content_length: usize = match cabecalhos.get("content-length") {
        Some(v) => v.parse().map_err(|_| ErroRequisicao::Invalida)?,
        None => 0,
    };
    if content_length > limite_corpo {
        return Err(ErroRequisicao::CorpoGrande);
    }

    let mut corpo = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut corpo)
            .map_err(|_| ErroRequisicao::Invalida)?;
    }

    Ok(Requisicao {
        metodo,
        caminho_url,
        query_str,
        cabecalhos,
        corpo,
    })
}

// -- Tratamento de requisicao --------------------------------------------------

fn tratar(stream: TcpStream, raiz: &Path) {
    let inicio = Instant::now();
    let timeout = std::env::var("PEP_REQUEST_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(15);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(timeout)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(timeout)));

    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return,
    };

    let req = match ler_requisicao(&stream) {
        Ok(r) => r,
        Err(erro) => {
            let (codigo, mensagem) = match erro {
                ErroRequisicao::Invalida => (400, "Requisicao invalida"),
                ErroRequisicao::CabecalhosGrandes => (431, "Cabecalhos muito grandes"),
                ErroRequisicao::CorpoGrande => (413, "Corpo da requisicao muito grande"),
                ErroRequisicao::CodificacaoNaoSuportada => (501, "Transfer-Encoding nao suportado"),
            };
            escrever_resposta(
                &mut writer,
                codigo,
                "text/plain; charset=utf-8",
                mensagem.as_bytes(),
                &[],
                true,
            );
            return;
        }
    };

    if req.caminho_url.contains("..") || req.caminho_url.contains('\0') {
        let corpo = pagina_erro(403, &req.caminho_url).into_bytes();
        escrever_resposta(
            &mut writer,
            403,
            "text/html; charset=utf-8",
            &corpo,
            &[],
            req.metodo != "HEAD",
        );
        return;
    }

    // ── WebSocket upgrade ──────────────────────────────────────────────────
    let is_ws = req
        .cabecalhos
        .get("upgrade")
        .map(|v| v.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false);
    let ws_key = req
        .cabecalhos
        .get("sec-websocket-key")
        .cloned()
        .unwrap_or_default();

    if is_ws && !ws_key.is_empty() {
        if let Some((handler, params)) = encontrar_rota(&req.metodo, &req.caminho_url) {
            let session_id = obter_ou_criar_sessao(&req);

            // Move o stream TCP para o módulo WS antes de chamar o handler
            let Some(guarda) = reservar_websocket() else {
                escrever_resposta(
                    &mut writer,
                    503,
                    "text/plain; charset=utf-8",
                    b"Limite de WebSockets atingido",
                    &[],
                    true,
                );
                return;
            };
            let handler = HandlerEnviavel(handler);
            std::thread::spawn(move || {
                let _guarda = guarda;
                crate::websocket::definir_stream_http(stream, ws_key);
                iniciar_contexto_servidor();
                crate::sessoes::definir_sessao_atual(session_id.clone());
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut vm = Maquina::nova();
                    vm.definir_globais(construir_contexto_http(&req, &session_id, params));
                    vm.chamar_funcao(handler.0, vec![])
                }));
                crate::websocket::limpar_thread();
                println!("WS {} fechado", req.caminho_url);
            });
            return;
        }
        // Tenta páginas PHP-style para WS (pages/ws/*.pep)
        if let Some((arquivo, params)) = encontrar_pagina(&req.caminho_url) {
            let session_id = obter_ou_criar_sessao(&req);
            let Some(guarda) = reservar_websocket() else {
                escrever_resposta(
                    &mut writer,
                    503,
                    "text/plain; charset=utf-8",
                    b"Limite de WebSockets atingido",
                    &[],
                    true,
                );
                return;
            };
            std::thread::spawn(move || {
                let _guarda = guarda;
                crate::websocket::definir_stream_http(stream, ws_key);
                executar_pep_em_processo(&arquivo, &req, &session_id, params);
                crate::websocket::limpar_thread();
                println!("WS {} fechado", req.caminho_url);
            });
            return;
        }
        let corpo = pagina_erro(404, &req.caminho_url).into_bytes();
        escrever_resposta(
            &mut writer,
            404,
            "text/html; charset=utf-8",
            &corpo,
            &[],
            req.metodo != "HEAD",
        );
        return;
    }

    // ── Roteamento HTTP normal ─────────────────────────────────────────────
    let metodos_permitidos = metodos_rota(&req.caminho_url);
    if req.metodo == "OPTIONS" && !metodos_permitidos.is_empty() {
        let extras = vec![("Allow".to_string(), metodos_permitidos.join(", "))];
        escrever_resposta(
            &mut writer,
            204,
            "text/plain; charset=utf-8",
            &[],
            &extras,
            false,
        );
        return;
    }
    let metodo_busca = if req.metodo == "HEAD" {
        "GET"
    } else {
        req.metodo.as_str()
    };
    if let Some((handler, params)) = encontrar_rota(metodo_busca, &req.caminho_url) {
        let session_id = obter_ou_criar_sessao(&req);

        // ── SSE: detecta Accept: text/event-stream ────────────────────────
        let aceita_sse = req
            .cabecalhos
            .get("accept")
            .map(|v| v.contains("text/event-stream"))
            .unwrap_or(false);

        if aceita_sse {
            // Registra o stream TCP no thread-local SSE, depois chama o handler.
            // O handler usa sse_iniciar/enviar/fechar. Não usamos escrever_resposta.
            let stream_clone = match stream.try_clone() {
                Ok(s) => s,
                Err(_) => {
                    escrever_resposta(
                        &mut writer,
                        500,
                        "text/plain; charset=utf-8",
                        b"Erro SSE",
                        &[],
                        true,
                    );
                    return;
                }
            };
            sse_definir_stream(stream_clone);
            iniciar_contexto_servidor();
            crate::sessoes::definir_sessao_atual(session_id.clone());
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mws: Vec<VmValor> = middlewares().read().unwrap().clone();
                let ctx = construir_contexto_http(&req, &session_id, params.clone());
                for mw in mws {
                    limpar_proximo();
                    let mut vm = Maquina::nova();
                    vm.definir_globais(ctx.clone());
                    let _ = vm.chamar_funcao(mw, vec![]);
                    if !proximo_foi_chamado() {
                        return;
                    }
                }
                let mut vm = Maquina::nova();
                vm.definir_globais(ctx);
                let _ = vm.chamar_funcao(handler, vec![]);
            }));
            sse_fechar();
            println!(
                "{} {} -> SSE encerrado ({} ms)",
                req.metodo,
                req.caminho_url,
                inicio.elapsed().as_millis()
            );
            return;
        }

        // ── Handler HTTP normal ───────────────────────────────────────────
        let mut extras: Vec<(String, String)> =
            vec![("Set-Cookie".to_string(), cookie_sessao(&req, &session_id))];

        let (codigo, hdrs, corpo) = executar_handler(handler, &req, &session_id, params);
        let mut ct = "text/html; charset=utf-8".to_string();
        for (k, v) in hdrs {
            if k.eq_ignore_ascii_case("content-type") {
                ct = v;
            } else {
                extras.push((k, v));
            }
        }
        escrever_resposta(
            &mut writer,
            codigo,
            &ct,
            &corpo,
            &extras,
            req.metodo != "HEAD",
        );
        println!(
            "{} {} -> {} (rota) ({} ms)",
            req.metodo,
            req.caminho_url,
            codigo,
            inicio.elapsed().as_millis()
        );
        return;
    }
    if !metodos_permitidos.is_empty() {
        let extras = vec![("Allow".to_string(), metodos_permitidos.join(", "))];
        escrever_resposta(
            &mut writer,
            405,
            "text/plain; charset=utf-8",
            b"Metodo nao permitido",
            &extras,
            req.metodo != "HEAD",
        );
        return;
    }

    // ── Páginas estilo PHP (pages/*.pep) ──────────────────────────────────────
    if let Some((arquivo, params)) = encontrar_pagina(&req.caminho_url) {
        if req.metodo == "OPTIONS" {
            let extras = vec![(
                "Allow".to_string(),
                "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS".to_string(),
            )];
            escrever_resposta(
                &mut writer,
                204,
                "text/plain; charset=utf-8",
                &[],
                &extras,
                false,
            );
            return;
        }
        let session_id = obter_ou_criar_sessao(&req);
        let mut extras: Vec<(String, String)> =
            vec![("Set-Cookie".to_string(), cookie_sessao(&req, &session_id))];
        let (codigo, hdrs, corpo) = executar_pep_em_processo(&arquivo, &req, &session_id, params);
        let mut ct = "text/html; charset=utf-8".to_string();
        for (k, v) in hdrs {
            if k.eq_ignore_ascii_case("content-type") {
                ct = v;
            } else {
                extras.push((k, v));
            }
        }
        escrever_resposta(
            &mut writer,
            codigo,
            &ct,
            &corpo,
            &extras,
            req.metodo != "HEAD",
        );
        println!(
            "{} {} -> {} (pagina) ({} ms)",
            req.metodo,
            req.caminho_url,
            codigo,
            inicio.elapsed().as_millis()
        );
        return;
    }

    let (codigo, content_type, corpo) = match resolver_arquivo_publico(&req.caminho_url, raiz) {
        None => (
            404u16,
            "text/html; charset=utf-8".to_string(),
            pagina_erro(404, &req.caminho_url).into_bytes(),
        ),
        Some(arquivo) => {
            let ext = arquivo.extension().and_then(|e| e.to_str()).unwrap_or("");
            match ext {
                "phtml" | "pep" => (
                    404,
                    "text/plain; charset=utf-8".to_string(),
                    b"Nao encontrado".to_vec(),
                ),
                _ => match std::fs::read(&arquivo) {
                    Ok(bytes) => (200, detectar_mime(ext).to_string(), bytes),
                    Err(e) => (
                        500,
                        "text/plain; charset=utf-8".to_string(),
                        e.to_string().into_bytes(),
                    ),
                },
            }
        }
    };

    escrever_resposta(
        &mut writer,
        codigo,
        &content_type,
        &corpo,
        &[],
        req.metodo != "HEAD",
    );
    println!(
        "{} {} -> {} ({} ms)",
        req.metodo,
        req.caminho_url,
        codigo,
        inicio.elapsed().as_millis()
    );
}

// -- Execucao in-process -------------------------------------------------------

fn executar_handler(
    handler: VmValor,
    req: &Requisicao,
    session_id: &str,
    params: HashMap<String, String>,
) -> (u16, Vec<(String, String)>, Vec<u8>) {
    iniciar_contexto_servidor();
    crate::sessoes::definir_sessao_atual(session_id.to_string());

    let ctx = construir_contexto_http(req, session_id, params);

    let resultado = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // ── Cadeia de middlewares ─────────────────────────────────────────────
        let mws: Vec<VmValor> = middlewares().read().unwrap().clone();
        for mw in mws {
            limpar_proximo();
            let mut vm = Maquina::nova();
            vm.definir_globais(ctx.clone());
            vm.chamar_funcao(mw, vec![])?;
            if !proximo_foi_chamado() {
                // Middleware respondeu sem chamar proximo() — interrompe a cadeia
                return Ok(VmValor::Null);
            }
        }
        // ── Handler da rota ───────────────────────────────────────────────────
        let mut vm = Maquina::nova();
        vm.definir_globais(ctx);
        vm.chamar_funcao(handler, vec![])
    }));

    let (status, hdrs, mut corpo) = drenar_resposta();
    match resultado {
        Ok(Ok(valor)) => {
            preencher_corpo_com_retorno(&mut corpo, valor);
            (status, hdrs, corpo)
        }
        Ok(Err(e)) => {
            if corpo.is_empty() {
                (500, vec![], pagina_erro_pep(&e).into_bytes())
            } else {
                (500, hdrs, corpo)
            }
        }
        Err(_) => (
            500,
            vec![],
            pagina_erro_pep("Erro interno (panic)").into_bytes(),
        ),
    }
}

fn construir_contexto_http(
    req: &Requisicao,
    session_id: &str,
    params: HashMap<String, String>,
) -> HashMap<Arc<str>, VmValor> {
    let mut g: HashMap<Arc<str>, VmValor> = HashMap::new();

    let str_mapa = |m: HashMap<String, String>| -> VmValor {
        VmValor::Mapa(m.into_iter().map(|(k, v)| (k, VmValor::Str(v))).collect())
    };

    g.insert(Arc::from("_GET"), str_mapa(parsear_query(&req.query_str)));

    let content_type = req
        .cabecalhos
        .get("content-type")
        .map(|s| s.as_str())
        .unwrap_or("");

    g.insert(
        Arc::from("_CORPO_BRUTO"),
        VmValor::Bytes(Arc::new(req.corpo.clone())),
    );

    let post_str = String::from_utf8_lossy(&req.corpo).into_owned();
    let mut arquivos: HashMap<String, VmValor> = HashMap::new();
    if content_type.contains("application/x-www-form-urlencoded") || content_type.is_empty() {
        g.insert(Arc::from("_POST"), str_mapa(parsear_query(&post_str)));
    } else if content_type.contains("multipart/form-data") {
        let (campos, uploads) = parsear_multipart(&req.corpo, content_type);
        g.insert(Arc::from("_POST"), VmValor::Mapa(campos));
        arquivos = uploads;
    } else {
        g.insert(Arc::from("_POST"), VmValor::Mapa(HashMap::new()));
    }
    g.insert(Arc::from("_ARQUIVOS"), VmValor::Mapa(arquivos));

    let corpo_json = if content_type.contains("application/json") {
        crate::interpretador::json_deserializar_val(&post_str)
            .map(valor_para_vm)
            .unwrap_or(VmValor::Null)
    } else {
        VmValor::Null
    };
    g.insert(Arc::from("_CORPO_JSON"), corpo_json);

    g.insert(Arc::from("_URL"), VmValor::Str(req.caminho_url.clone()));
    g.insert(Arc::from("_METODO"), VmValor::Str(req.metodo.clone()));
    g.insert(
        Arc::from("_SESSAO_ID"),
        VmValor::Str(session_id.to_string()),
    );
    g.insert(Arc::from("_SERVIDOR"), VmValor::Bool(true));

    let cookie_str = req.cabecalhos.get("cookie").cloned().unwrap_or_default();
    let cookies: HashMap<String, VmValor> = cookie_str
        .split(';')
        .filter_map(|c| {
            c.trim()
                .split_once('=')
                .map(|(k, v)| (k.trim().to_string(), VmValor::Str(v.trim().to_string())))
        })
        .collect();
    g.insert(Arc::from("_COOKIE"), VmValor::Mapa(cookies));

    let headers: HashMap<String, VmValor> = req
        .cabecalhos
        .iter()
        .map(|(k, v)| (k.clone(), VmValor::Str(v.clone())))
        .collect();
    g.insert(Arc::from("_CABECALHOS"), VmValor::Mapa(headers));

    let mut requisicao: HashMap<String, VmValor> = HashMap::new();
    requisicao.insert("metodo".to_string(), VmValor::Str(req.metodo.clone()));
    requisicao.insert("url".to_string(), VmValor::Str(req.caminho_url.clone()));
    requisicao.insert("query".to_string(), VmValor::Str(req.query_str.clone()));
    g.insert(Arc::from("_REQUISICAO"), VmValor::Mapa(requisicao));

    let params_map: HashMap<String, VmValor> = params
        .into_iter()
        .map(|(k, v)| (k, VmValor::Str(v)))
        .collect();
    g.insert(Arc::from("_PARAMS"), VmValor::Mapa(params_map));

    g
}

fn executar_pep_em_processo(
    arquivo: &Path,
    req: &Requisicao,
    session_id: &str,
    params: HashMap<String, String>,
) -> (u16, Vec<(String, String)>, Vec<u8>) {
    iniciar_contexto_servidor();
    crate::sessoes::definir_sessao_atual(session_id.to_string());

    let resultado = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ops = compilar_arquivo_bytecode(arquivo)?;
        let mut vm = Maquina::com_base(
            arquivo
                .parent()
                .map(|d| d.to_path_buf())
                .unwrap_or_default(),
        );
        vm.definir_globais(construir_contexto_http(req, session_id, params));
        vm.executar_com_retorno(&ops)
    }));

    let (status, hdrs, mut corpo) = drenar_resposta();

    match resultado {
        Ok(Ok(valor)) => {
            preencher_corpo_com_retorno(&mut corpo, valor);
            (status, hdrs, corpo)
        }
        Ok(Err(e)) => {
            if corpo.is_empty() {
                (500, vec![], pagina_erro_pep(&e).into_bytes())
            } else {
                (500, hdrs, corpo)
            }
        }
        Err(_panic) => (
            500,
            vec![],
            pagina_erro_pep("Erro interno (panic)").into_bytes(),
        ),
    }
}

fn preencher_corpo_com_retorno(corpo: &mut Vec<u8>, valor: VmValor) {
    if !corpo.is_empty() {
        return;
    }
    match valor {
        VmValor::Null => {}
        VmValor::Bytes(bytes) => corpo.extend_from_slice(bytes.as_ref()),
        VmValor::Str(texto) => corpo.extend_from_slice(texto.as_bytes()),
        outro => corpo.extend_from_slice(outro.to_string().as_bytes()),
    }
}

fn compilar_arquivo_bytecode(arquivo: &Path) -> Result<Vec<crate::bytecode::Op>, String> {
    let modificado = std::fs::metadata(arquivo)
        .and_then(|m| m.modified())
        .map_err(|e| format!("Erro ao examinar '{}': {}", arquivo.display(), e))?;
    if let Some((quando, ops)) = cache_bytecode().read().unwrap().get(arquivo) {
        if *quando == modificado {
            return Ok(ops.clone());
        }
    }
    let fonte = std::fs::read_to_string(arquivo)
        .map_err(|e| format!("Erro ao ler '{}': {}", arquivo.display(), e))?;
    let ext = arquivo.extension().and_then(|e| e.to_str()).unwrap_or("");
    let e_template = matches!(ext, "phtml" | "html" | "htm");
    let ops = compilar_bytecode_str(&fonte, e_template)?;
    cache_bytecode()
        .write()
        .unwrap()
        .insert(arquivo.to_path_buf(), (modificado, ops.clone()));
    Ok(ops)
}

fn compilar_bytecode_fonte(fonte: &str, _base: &Path) -> Result<Vec<crate::bytecode::Op>, String> {
    compilar_bytecode_str(fonte, false)
}

fn compilar_bytecode_str(
    fonte: &str,
    e_template: bool,
) -> Result<Vec<crate::bytecode::Op>, String> {
    let programa = if e_template {
        crate::template::compilar(fonte)?
    } else {
        let mut lex = crate::lexer::Lexer::novo(fonte);
        let tokens = lex.tokenizar()?;
        crate::parser::Parser::novo(tokens).parsear()?
    };
    crate::compilador::compilar(&programa)
}

/// Interface pública para o modo FastCGI.
/// Retorna (status, cabecalhos_extras, corpo).
pub fn executar_pep_fastcgi(
    script_ou_raiz: &str,
    metodo: String,
    caminho_url: String,
    query_str: String,
    cabecalhos: HashMap<String, String>,
    corpo_req: Vec<u8>,
) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let cabecalhos = cabecalhos
        .into_iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v))
        .collect();
    let req = Requisicao {
        metodo,
        caminho_url,
        query_str,
        cabecalhos,
        corpo: corpo_req,
    };
    let alvo = Path::new(script_ou_raiz);
    if alvo.is_dir() {
        let permitidos = metodos_rota(&req.caminho_url);
        if req.metodo == "OPTIONS" && !permitidos.is_empty() {
            return (
                204,
                vec![("Allow".to_string(), permitidos.join(", "))],
                Vec::new(),
            );
        }
        let metodo_busca = if req.metodo == "HEAD" {
            "GET"
        } else {
            req.metodo.as_str()
        };
        if let Some((handler, params)) = encontrar_rota(metodo_busca, &req.caminho_url) {
            let session_id = obter_ou_criar_sessao(&req);
            let (status, mut headers, corpo) = executar_handler(handler, &req, &session_id, params);
            headers.push(("Set-Cookie".to_string(), cookie_sessao(&req, &session_id)));
            return (
                status,
                headers,
                if req.metodo == "HEAD" {
                    Vec::new()
                } else {
                    corpo
                },
            );
        }
        if !permitidos.is_empty() {
            return (
                405,
                vec![("Allow".to_string(), permitidos.join(", "))],
                b"Metodo nao permitido".to_vec(),
            );
        }
    }
    let (arquivo, params) = if alvo.is_dir() {
        let pagina = resolver_script_na_raiz(alvo, &req.caminho_url).or_else(|| {
            descobrir_paginas(&alvo.join("pages"))
                .into_iter()
                .find_map(|(padrao, arquivo)| {
                    combinar_padrao(&padrao, &req.caminho_url).map(|params| (arquivo, params))
                })
        });
        match pagina {
            Some(v) => {
                if req.metodo == "OPTIONS" {
                    return (
                        204,
                        vec![(
                            "Allow".to_string(),
                            "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS".to_string(),
                        )],
                        Vec::new(),
                    );
                }
                v
            }
            None => {
                return (
                    404,
                    vec![(
                        "Content-Type".to_string(),
                        "text/plain; charset=utf-8".to_string(),
                    )],
                    b"Nao encontrado".to_vec(),
                )
            }
        }
    } else {
        (alvo.to_path_buf(), HashMap::new())
    };
    let session_id = obter_ou_criar_sessao(&req);
    let (status, mut headers, corpo) =
        executar_pep_em_processo(&arquivo, &req, &session_id, params);
    headers.push(("Set-Cookie".to_string(), cookie_sessao(&req, &session_id)));
    (
        status,
        headers,
        if req.metodo == "HEAD" {
            Vec::new()
        } else {
            corpo
        },
    )
}

// -- HTTP utilitarios ----------------------------------------------------------

fn escrever_resposta(
    writer: &mut impl Write,
    codigo: u16,
    content_type: &str,
    corpo: &[u8],
    extras: &[(String, String)],
    enviar_corpo: bool,
) {
    let mut cab = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        codigo,
        texto_status(codigo),
        content_type,
        corpo.len()
    );
    for (k, v) in extras {
        cab.push_str(&format!("{}: {}\r\n", k, v));
    }
    cab.push_str("\r\n");
    let _ = writer.write_all(cab.as_bytes());
    if enviar_corpo {
        let _ = writer.write_all(corpo);
    }
}

// -- Resolucao de arquivos -----------------------------------------------------

fn resolver_arquivo_publico(url: &str, raiz: &Path) -> Option<PathBuf> {
    let publico = raiz.join("public").canonicalize().ok()?;
    let decodificado = url_decodificar(url);
    let relativo = decodificado
        .trim_start_matches('/')
        .strip_prefix("public/")
        .unwrap_or_else(|| decodificado.trim_start_matches('/'));
    let caminho = Path::new(relativo);
    if caminho.components().any(|c| {
        matches!(
            c,
            std::path::Component::ParentDir
                | std::path::Component::RootDir
                | std::path::Component::Prefix(_)
        )
    }) {
        return None;
    }
    let candidato = publico.join(caminho);

    if candidato.is_dir() || relativo.is_empty() {
        let dir = if relativo.is_empty() {
            publico.clone()
        } else {
            candidato
        };
        for nome in &["index.html", "index.htm"] {
            let idx = dir.join(nome);
            if let Ok(real) = idx.canonicalize() {
                if real.starts_with(&publico) {
                    return Some(real);
                }
            }
        }
        None
    } else if let Ok(real) = candidato.canonicalize() {
        real.starts_with(&publico).then_some(real)
    } else {
        None
    }
}

// -- Query string e cookies ----------------------------------------------------

pub fn parsear_query(query: &str) -> HashMap<String, String> {
    let mut mapa = HashMap::new();
    if query.is_empty() {
        return mapa;
    }
    for par in query.split('&') {
        if let Some((k, v)) = par.split_once('=') {
            mapa.insert(url_decodificar(k), url_decodificar(v));
        } else if !par.is_empty() {
            mapa.insert(url_decodificar(par), String::new());
        }
    }
    mapa
}

fn parsear_multipart(
    corpo: &[u8],
    content_type: &str,
) -> (HashMap<String, VmValor>, HashMap<String, VmValor>) {
    let mut campos: HashMap<String, VmValor> = HashMap::new();
    let mut arquivos: HashMap<String, VmValor> = HashMap::new();
    let Some(boundary) = content_type.split(';').find_map(|parte| {
        parte
            .trim()
            .strip_prefix("boundary=")
            .map(|v| v.trim_matches('"').to_string())
    }) else {
        return (campos, arquivos);
    };
    if boundary.is_empty() || boundary.len() > 200 {
        return (campos, arquivos);
    }
    let delimitador = format!("--{}", boundary).into_bytes();
    let mut cursor = 0usize;
    while let Some(inicio_rel) = encontrar_bytes(&corpo[cursor..], &delimitador) {
        let mut inicio = cursor + inicio_rel + delimitador.len();
        if corpo.get(inicio..inicio + 2) == Some(b"--") {
            break;
        }
        if corpo.get(inicio..inicio + 2) == Some(b"\r\n") {
            inicio += 2;
        }
        let Some(fim_rel) = encontrar_bytes(&corpo[inicio..], &delimitador) else {
            break;
        };
        let mut parte = &corpo[inicio..inicio + fim_rel];
        if parte.ends_with(b"\r\n") {
            parte = &parte[..parte.len() - 2];
        }
        cursor = inicio + fim_rel;
        let Some(separador) = encontrar_bytes(parte, b"\r\n\r\n") else {
            continue;
        };
        let cabecalhos = String::from_utf8_lossy(&parte[..separador]);
        let dados = &parte[separador + 4..];
        let disposicao = cabecalhos
            .lines()
            .find(|l| l.to_ascii_lowercase().starts_with("content-disposition:"))
            .unwrap_or("");
        let Some(nome) = parametro_cabecalho(disposicao, "name") else {
            continue;
        };
        if let Some(arquivo) = parametro_cabecalho(disposicao, "filename") {
            let tipo = cabecalhos
                .lines()
                .find_map(|l| {
                    l.split_once(':')
                        .filter(|(k, _)| k.eq_ignore_ascii_case("content-type"))
                })
                .map(|(_, v)| v.trim())
                .unwrap_or("application/octet-stream");
            let mut upload: HashMap<String, VmValor> = HashMap::new();
            upload.insert("nome".to_string(), VmValor::Str(arquivo));
            upload.insert("tipo".to_string(), VmValor::Str(tipo.to_string()));
            upload.insert("tamanho".to_string(), VmValor::Int(dados.len() as i64));
            upload.insert(
                "dados".to_string(),
                VmValor::Bytes(Arc::new(dados.to_vec())),
            );
            arquivos.insert(nome, VmValor::Mapa(upload));
        } else {
            campos.insert(
                nome,
                VmValor::Str(String::from_utf8_lossy(dados).into_owned()),
            );
        }
    }
    (campos, arquivos)
}

fn encontrar_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|janela| janela == needle)
}

fn parametro_cabecalho(cabecalho: &str, nome: &str) -> Option<String> {
    cabecalho.split(';').skip(1).find_map(|parte| {
        let (chave, valor) = parte.trim().split_once('=')?;
        chave
            .eq_ignore_ascii_case(nome)
            .then(|| valor.trim().trim_matches('"').to_string())
    })
}

pub fn url_decodificar(s: &str) -> String {
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

fn extrair_cookie(header: &str, nome: &str) -> Option<String> {
    for par in header.split(';') {
        let par = par.trim();
        if let Some((k, v)) = par.split_once('=') {
            if k.trim() == nome {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

fn id_sessao_valido(id: &str) -> bool {
    id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit())
}

fn nova_sessao_id() -> String {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).expect("gerador aleatorio seguro indisponivel");
    hex::encode(bytes)
}

fn obter_ou_criar_sessao(req: &Requisicao) -> String {
    req.cabecalhos
        .get("cookie")
        .and_then(|c| extrair_cookie(c, "pep_sessao"))
        .filter(|id| id_sessao_valido(id))
        .unwrap_or_else(nova_sessao_id)
}

fn cookie_sessao(req: &Requisicao, id: &str) -> String {
    let seguro = std::env::var("PEP_COOKIE_SECURE")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        || req
            .cabecalhos
            .get("x-forwarded-proto")
            .is_some_and(|v| v.eq_ignore_ascii_case("https"));
    format!(
        "pep_sessao={}; Path=/; HttpOnly; SameSite=Lax{}",
        id,
        if seguro { "; Secure" } else { "" }
    )
}

// -- Tipos MIME ----------------------------------------------------------------

fn detectar_mime(ext: &str) -> &'static str {
    match ext {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" => "application/javascript; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "txt" => "text/plain; charset=utf-8",
        "xml" => "application/xml; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        "webp" => "image/webp",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
}

// -- Paginas de erro HTML ------------------------------------------------------

pub(crate) fn texto_status(codigo: u16) -> &'static str {
    match codigo {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        206 => "Partial Content",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        413 => "Payload Too Large",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Unknown",
    }
}

fn pagina_erro(codigo: u16, caminho: &str) -> String {
    format!(
        r#"<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
<title>{0} {1}</title>
<style>body{{font-family:sans-serif;max-width:600px;margin:80px auto;color:#333}}h1{{color:#c00}}</style>
</head><body><h1>{0} {1}</h1><p>Nao encontrado: <code>{2}</code></p></body></html>"#,
        codigo,
        texto_status(codigo),
        caminho
    )
}

fn pagina_erro_pep(erro: &str) -> String {
    let ee = erro
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    format!(
        r#"<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
<title>500 - Erro PEP</title>
<style>body{{font-family:sans-serif;max-width:800px;margin:40px auto;color:#333}}h1{{color:#c00}}
pre{{background:#fff3f3;border:1px solid #f99;padding:16px;border-radius:4px;white-space:pre-wrap}}</style>
</head><body><h1>Erro ao executar script PEP</h1><pre>{}</pre></body></html>"#,
        ee
    )
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn retorno_texto_preenche_corpo_vazio() {
        let mut corpo = Vec::new();
        preencher_corpo_com_retorno(&mut corpo, VmValor::Str("ola".to_string()));
        assert_eq!(corpo, b"ola");
    }

    #[test]
    fn multipart_separa_campos_e_arquivos() {
        let corpo = b"--abc\r\nContent-Disposition: form-data; name=\"titulo\"\r\n\r\nOi\r\n--abc\r\nContent-Disposition: form-data; name=\"foto\"; filename=\"a.txt\"\r\nContent-Type: text/plain\r\n\r\ndados\r\n--abc--\r\n";
        let (campos, arquivos) = parsear_multipart(corpo, "multipart/form-data; boundary=abc");
        assert_eq!(campos.get("titulo"), Some(&VmValor::Str("Oi".to_string())));
        assert!(matches!(arquivos.get("foto"), Some(VmValor::Mapa(_))));
    }

    #[test]
    fn somente_public_e_resolvido_como_estatico() {
        let raiz = std::env::temp_dir().join(format!("pep-public-{}", nova_sessao_id()));
        std::fs::create_dir_all(raiz.join("public")).unwrap();
        std::fs::write(raiz.join("public/app.js"), b"ok").unwrap();
        std::fs::write(raiz.join("segredo.txt"), b"nao").unwrap();
        assert!(resolver_arquivo_publico("/public/app.js", &raiz).is_some());
        assert!(resolver_arquivo_publico("/segredo.txt", &raiz).is_none());
        let _ = std::fs::remove_dir_all(raiz);
    }

    #[test]
    fn sessao_exige_256_bits_em_hex() {
        assert!(id_sessao_valido(&"a".repeat(64)));
        assert!(!id_sessao_valido("abc"));
        assert!(!id_sessao_valido(&"z".repeat(64)));
    }

    #[test]
    fn roteamento_por_arquivos_prioriza_rota_fixa() {
        let raiz = std::env::temp_dir().join(format!("pep-pages-{}", nova_sessao_id()));
        let pages = raiz.join("pages/usuarios");
        std::fs::create_dir_all(&pages).unwrap();
        std::fs::write(pages.join("novo.pep"), b"retornar \"novo\"").unwrap();
        std::fs::write(pages.join("[id].pep"), b"retornar _PARAMS[\"id\"]").unwrap();
        let rotas = descobrir_paginas(&raiz.join("pages"));
        assert_eq!(rotas[0].0, "/usuarios/novo");
        assert_eq!(
            combinar_padrao(&rotas[1].0, "/usuarios/42")
                .unwrap()
                .get("id")
                .map(String::as_str),
            Some("42")
        );
        let _ = std::fs::remove_dir_all(raiz);
    }

    #[test]
    fn raiz_documental_funciona_como_php() {
        let raiz = std::env::temp_dir().join(format!("pep-docroot-{}", nova_sessao_id()));
        std::fs::create_dir_all(raiz.join("produto")).unwrap();
        std::fs::create_dir_all(raiz.join("blog")).unwrap();
        std::fs::write(raiz.join("index.pep"), b"retornar \"inicio\"").unwrap();
        std::fs::write(raiz.join("produto/[id].pep"), b"retornar _PARAMS[\"id\"]").unwrap();
        std::fs::write(raiz.join("blog/index.phtml"), b"blog").unwrap();

        assert!(resolver_script_na_raiz(&raiz, "/")
            .unwrap()
            .0
            .ends_with("index.pep"));
        let (_, params) = resolver_script_na_raiz(&raiz, "/produto/42").unwrap();
        assert_eq!(params.get("id").map(String::as_str), Some("42"));
        assert!(resolver_script_na_raiz(&raiz, "/blog")
            .unwrap()
            .0
            .ends_with("index.phtml"));
        assert!(resolver_script_na_raiz(&raiz, "/index.pep").is_some());
        let _ = std::fs::remove_dir_all(raiz);
    }
}
