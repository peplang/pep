use pep::ast::Instrucao;
use pep::interpretador::{Interpretador, Valor};
use pep::lexer::Lexer;
use pep::parser::Parser;
use pep::{compilador, fastcgi, lsp, register_vm, servidor, template, vm};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(target_os = "windows")]
fn configurar_console_utf8() {
    extern "system" {
        fn SetConsoleOutputCP(wCodePageID: u32) -> i32;
        fn SetConsoleCP(wCodePageID: u32) -> i32;
    }
    unsafe {
        SetConsoleOutputCP(65001);
        SetConsoleCP(65001);
    }
}

#[cfg(not(target_os = "windows"))]
fn configurar_console_utf8() {}

const VERSAO: &str = "0.6.0";

fn executar_codigo_com_base(
    codigo: &str,
    interp: &mut Interpretador,
    base_import: Option<&Path>,
) -> Result<(), String> {
    let mut lexer = Lexer::novo(codigo);
    let tokens = lexer.tokenizar()?;
    let mut parser = Parser::novo(tokens);
    let programa = parser.parsear()?;
    if let Some(base) = base_import {
        interp.entrar_diretorio_importacao(base.to_path_buf());
    }
    let resultado = interp.executar(&programa);
    if base_import.is_some() {
        interp.sair_diretorio_importacao();
    }
    resultado
}

fn executar_codigo(codigo: &str, interp: &mut Interpretador) -> Result<(), String> {
    executar_codigo_com_base(codigo, interp, None)
}

fn processar_template_com_base(
    fonte: &str,
    interp: &mut Interpretador,
    base_import: Option<&Path>,
) -> Result<(), String> {
    if let Some(base) = base_import {
        interp.entrar_diretorio_importacao(base.to_path_buf());
    }
    let resultado = template::processar(fonte, interp);
    if base_import.is_some() {
        interp.sair_diretorio_importacao();
    }
    resultado
}

fn verificar_codigo(codigo: &str) -> Result<(), String> {
    let mut lexer = Lexer::novo(codigo);
    let tokens = lexer.tokenizar()?;
    let mut parser = Parser::novo(tokens);
    parser.parsear()?;
    Ok(())
}

fn e_template(caminho: &str) -> bool {
    matches!(
        Path::new(caminho).extension().and_then(|e| e.to_str()),
        Some("phtml") | Some("html") | Some("htm")
    )
}

/// Injeta variaveis de contexto HTTP no interpretador (usadas em templates/scripts servidos)
fn injetar_contexto_servidor(interp: &mut Interpretador) {
    let em_servidor = std::env::var("PEP_SERVIDOR").is_ok();
    interp
        .ambiente
        .definir("_SERVIDOR".to_string(), Valor::Booleano(em_servidor));

    let url = std::env::var("PEP_URL").unwrap_or_default();
    interp
        .ambiente
        .definir("_URL".to_string(), Valor::Texto(url));

    let metodo = std::env::var("PEP_METHOD").unwrap_or_else(|_| "GET".to_string());
    interp
        .ambiente
        .definir("_METODO".to_string(), Valor::Texto(metodo.clone()));

    // _GET  -  query string como mapa
    let query = std::env::var("PEP_QUERY_STRING").unwrap_or_default();
    let params_get = servidor::parsear_query(&query);
    let mapa_get: HashMap<String, Valor> = params_get
        .into_iter()
        .map(|(k, v)| (k, Valor::Texto(v)))
        .collect();
    interp
        .ambiente
        .definir("_GET".to_string(), Valor::Mapa(mapa_get));

    // _POST  -  corpo de requisicoes POST como mapa (application/x-www-form-urlencoded)
    let post_raw = std::env::var("PEP_POST_DATA").unwrap_or_default();
    let params_post = servidor::parsear_query(&post_raw);
    let mapa_post: HashMap<String, Valor> = params_post
        .into_iter()
        .map(|(k, v)| (k, Valor::Texto(v)))
        .collect();
    interp
        .ambiente
        .definir("_POST".to_string(), Valor::Mapa(mapa_post));

    let mut mapa_request = match interp.ambiente.obter("_GET") {
        Some(Valor::Mapa(m)) => m,
        _ => HashMap::new(),
    };
    if let Some(Valor::Mapa(post)) = interp.ambiente.obter("_POST") {
        for (k, v) in post {
            mapa_request.insert(k, v);
        }
    }
    interp
        .ambiente
        .definir("_REQUEST".to_string(), Valor::Mapa(mapa_request));

    // _COOKIE  -  cookies como mapa
    let cookie_str = std::env::var("PEP_COOKIE").unwrap_or_default();
    let mut mapa_cookie: HashMap<String, Valor> = HashMap::new();
    for par in cookie_str.split(';') {
        let par = par.trim();
        if let Some((k, v)) = par.split_once('=') {
            mapa_cookie.insert(
                k.trim().to_string(),
                Valor::Texto(servidor::url_decodificar(v.trim())),
            );
        }
    }
    interp
        .ambiente
        .definir("_COOKIE".to_string(), Valor::Mapa(mapa_cookie));

    let mut mapa_server: HashMap<String, Valor> = HashMap::new();
    mapa_server.insert("REQUEST_METHOD".to_string(), Valor::Texto(metodo));
    mapa_server.insert(
        "REQUEST_URI".to_string(),
        interp
            .ambiente
            .obter("_URL")
            .unwrap_or(Valor::Texto(String::new())),
    );
    mapa_server.insert("QUERY_STRING".to_string(), Valor::Texto(query));
    mapa_server.insert("HTTP_COOKIE".to_string(), Valor::Texto(cookie_str));
    interp
        .ambiente
        .definir("_SERVER".to_string(), Valor::Mapa(mapa_server));
}

fn contar_delimitadores_fora_de_strings(s: &str) -> (usize, usize, usize, usize) {
    let mut chaves_ab = 0usize;
    let mut chaves_fe = 0usize;
    let mut parens_ab = 0usize;
    let mut parens_fe = 0usize;
    let mut em_string = false;
    let mut delimitador = '"';
    let mut escape = false;
    for c in s.chars() {
        if escape {
            escape = false;
            continue;
        }
        if em_string {
            if c == '\\' {
                escape = true;
            } else if c == delimitador {
                em_string = false;
            }
            continue;
        }
        match c {
            '"' | '\'' => {
                em_string = true;
                delimitador = c;
            }
            '{' => chaves_ab += 1,
            '}' => chaves_fe += 1,
            '(' => parens_ab += 1,
            ')' => parens_fe += 1,
            _ => {}
        }
    }
    (chaves_ab, chaves_fe, parens_ab, parens_fe)
}

fn repl() {
    println!("PEP - Programar em Portugues v{}", VERSAO);
    println!("Digite 'sair' para encerrar, 'ajuda' para ajuda.\n");

    let mut interp = Interpretador::novo();
    let mut buffer = String::new();
    let mut editor = match DefaultEditor::new() {
        Ok(editor) => editor,
        Err(e) => {
            eprintln!("Erro ao iniciar REPL: {}", e);
            return;
        }
    };
    let historico = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(|dir| PathBuf::from(dir).join(".pep_historico"));
    if let Some(path) = &historico {
        let _ = editor.load_history(path);
    }

    loop {
        let prompt = if buffer.is_empty() { "pep> " } else { "...  " };
        let linha = match editor.readline(prompt) {
            Ok(linha) => linha,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("Erro: {}", e);
                break;
            }
        };

        let linha_trim = linha.trim();
        match linha_trim {
            "sair" | "saida" => {
                println!("Ate logo!");
                break;
            }
            "ajuda" => {
                imprimir_ajuda();
                continue;
            }
            "" if buffer.is_empty() => continue,
            _ => {}
        }

        if !linha_trim.is_empty() {
            let _ = editor.add_history_entry(linha_trim);
        }
        buffer.push_str(&linha);
        buffer.push('\n');

        let (abertas, fechadas, par_ab, par_fe) = contar_delimitadores_fora_de_strings(&buffer);

        if abertas == fechadas && par_ab == par_fe {
            let codigo = buffer.trim().to_string();
            buffer.clear();
            if !codigo.is_empty() {
                if let Err(e) = executar_codigo(&codigo, &mut interp) {
                    eprintln!("Erro: {}", e);
                }
            }
        }
    }
    if let Some(path) = &historico {
        let _ = editor.save_history(path);
    }
}

fn formatar_codigo(codigo: &str) -> String {
    let mut nivel = 0usize;
    let mut saida = String::new();
    for linha in codigo.lines() {
        let texto = linha.trim();
        if texto.is_empty() {
            saida.push('\n');
            continue;
        }
        let fechamentos_iniciais = texto.chars().take_while(|c| *c == '}').count();
        nivel = nivel.saturating_sub(fechamentos_iniciais);
        saida.push_str(&"    ".repeat(nivel));
        saida.push_str(texto);
        saida.push('\n');
        let (abertas, fechadas, _, _) = contar_delimitadores_fora_de_strings(texto);
        nivel += abertas;
        nivel = nivel.saturating_sub(fechadas.saturating_sub(fechamentos_iniciais));
    }
    saida
}

fn gerar_documentacao(arquivo: &str, programa: &[Instrucao]) -> String {
    let mut doc = format!("# Documentacao de `{}`\n\n", arquivo);
    let mut encontrou = false;
    for instrucao in programa {
        let (linha, interna) = match instrucao {
            Instrucao::Localizada {
                linha, instrucao, ..
            } => (Some(*linha), instrucao.as_ref()),
            outra => (None, outra),
        };
        if let Instrucao::Funcao {
            nome, parametros, ..
        } = interna
        {
            encontrou = true;
            doc.push_str(&format!("## `{}`\n\n", nome));
            doc.push_str(&format!(
                "```pep\nfuncao {}({})\n```\n\n",
                nome,
                parametros.iter().map(|p| {
                    if p.variadic { format!("...{}", p.nome) }
                    else if p.padrao.is_some() { format!("{}=...", p.nome) }
                    else { p.nome.clone() }
                }).collect::<Vec<_>>().join(", ")
            ));
            if let Some(linha) = linha {
                doc.push_str(&format!("Definida na linha {}.\n\n", linha));
            }
        }
    }
    if !encontrou {
        doc.push_str("Nenhuma funcao publica encontrada.\n");
    }
    doc
}

fn imprimir_ajuda() {
    println!();
    println!("=== PEP - Programar em Portugues v{} ===", VERSAO);
    println!();
    println!("USO:");
    println!("  pep                           -  REPL interativo");
    println!("  pep arquivo.pep               -  executa arquivo PEP");
    println!("  pep arquivo.phtml             -  processa template PEP");
    println!("  pep --check arquivo          - verifica sintaxe sem executar");
    println!("  pep formatar arquivo.pep     - formata o codigo no arquivo");
    println!("  pep doc arquivo.pep          - gera documentacao Markdown");
    println!("  pep servir [porta] [dir]      -  servidor HTTP (padrao: 7878)");
    println!("  pep fastcgi <script|dir> [porta] - FastCGI (padrao: 9000)");
    println!("  pep --vm-reg arquivo.pep       - VM experimental baseada em registradores");
    println!("  pep lsp                        - servidor LSP via entrada/saida padrao");
    println!();
    println!("NOVIDADES v0.3:");
    println!("  funcao(x) {{ ... }}           - funcoes anonimas e closures");
    println!("  mapear/filtrar/reduzir       - funcoes de ordem superior");
    println!("  7 // 2                       - divisao inteira");
    println!("  pep_modules/                 - pacotes locais");
    println!("  mapa.campo                    - acesso por ponto em mapas");
    println!("  += -= *= /= %=               -  operadores compostos");
    println!("  \"Ola, {{nome}}!\"               -  interpolacao de strings");
    println!("  tentar {{ }} capturar (e) {{ }}   -  tratamento de erros");
    println!("  lancar \"mensagem\"             -  lancar erro");
    println!("  importar \"arquivo.pep\"        -  importar modulo");
    println!("  importar \"arquivo.pep\" como m - importa com namespace");
    println!("  incluir \"arquivo.phtml\"       - inclui arquivo no ambiente atual");
    println!("  requerer \"arquivo.pep\"        - inclui ou falha se nao existir");
    println!();
    println!("FUNCOES DE TEXTO:");
    println!("  maiusculas(t)  minusculas(t)  aparar(t)  substituir(t, de, para)");
    println!("  dividir(t, sep)  juntar(lista, sep)  sub_texto(t, ini, fim)");
    println!("  comeca_com(t, p)  termina_com(t, s)  contem_texto(t, s)  posicao(t, s)");
    println!("  html_escapar(t)  url_codificar(t)");
    println!();
    println!("FUNCOES MATEMATICAS:");
    println!("  raiz(n)  potencia(b, e)  absoluto(n)  arredondar(n, casas)");
    println!("  piso(n)  teto(n)  minimo(a, b)  maximo(a, b)");
    println!("  aleatorio()  aleatorio_inteiro(min, max)  pi()");
    println!("  seno(n)  cosseno(n)  tangente(n)  logaritmo(n)");
    println!();
    println!("FUNCOES DE ARQUIVO:");
    println!("  ler_arquivo(c)  escrever_arquivo(c, t)  acrescentar_arquivo(c, t)");
    println!("  arquivo_existe(c)  apagar_arquivo(c)  listar_arquivos(c)  criar_diretorio(c)");
    println!();
    println!("JSON:");
    println!("  json_serializar(v)  json_deserializar(t)");
    println!("  ffi_permitida()  ffi_carregar(lib)  ffi_chamar(id, simbolo, dados)  ffi_fechar(id)");
    println!();
    println!("WEB (em templates):");
    println!("  _GET  _POST  _ARQUIVOS  _COOKIE  _CABECALHOS  _URL  _METODO  _PARAMS");
    println!("  sessao_iniciar()  sessao_obter(k)  sessao_definir(k, v)  sessao_regenerar()");
    println!("  cookie_obter(nome)  cookie_definir(nome, valor, opcoes?)");
    println!("  cabecalho(nome, valor)  status(codigo)  redirecionar(url)");
    println!("  json_responder(valor)  entrada_get(nome)  entrada_post(nome)");
    println!();
    println!("BANCO DE DADOS:");
    println!("  bd_conectar(url)  bd_consultar(c, sql)  bd_executar(c, sql)  bd_fechar(c)");
    println!("  sqlite_conectar(caminho)  sqlite_consultar(c, sql, params)");
    println!("  sqlite_executar(c, sql, params)  sqlite_fechar(c)");
    println!("HTTP CLIENTE:");
    println!("  obter_url(url)  postar_url(url, dados)");
    println!();
}

fn main() {
    configurar_console_utf8();
    let args: Vec<String> = std::env::args().collect();

    match args.as_slice() {
        [_] => repl(),

        [_, flag] if flag == "--ajuda" || flag == "-a" => imprimir_ajuda(),
        [_, flag] if flag == "--versao" || flag == "--version" || flag == "-v" => {
            println!("PEP v{}", VERSAO)
        }
        [_, cmd] if cmd == "lsp" => {
            if let Err(e) = lsp::iniciar_stdio() {
                eprintln!("Erro no LSP: {}", e);
                std::process::exit(1);
            }
        }

        [_, cmd, arquivo] if cmd == "formatar" => {
            let fonte = fs::read_to_string(arquivo).unwrap_or_else(|e| {
                eprintln!("Erro ao abrir '{}': {}", arquivo, e);
                std::process::exit(1)
            });
            fs::write(arquivo, formatar_codigo(&fonte)).unwrap_or_else(|e| {
                eprintln!("Erro ao escrever '{}': {}", arquivo, e);
                std::process::exit(1)
            });
            println!("Formatado: {}", arquivo);
        }

        [_, cmd, arquivo] if cmd == "doc" => {
            let fonte = fs::read_to_string(arquivo).unwrap_or_else(|e| {
                eprintln!("Erro ao abrir '{}': {}", arquivo, e);
                std::process::exit(1)
            });
            let resultado = (|| -> Result<PathBuf, String> {
                let mut lexer = Lexer::novo(&fonte);
                let tokens = lexer.tokenizar()?;
                let mut parser = Parser::novo(tokens);
                let programa = parser.parsear()?;
                let destino = Path::new(arquivo).with_extension("md");
                fs::write(&destino, gerar_documentacao(arquivo, &programa))
                    .map_err(|e| e.to_string())?;
                Ok(destino)
            })();
            match resultado {
                Ok(destino) => println!("Documentacao: {}", destino.display()),
                Err(e) => {
                    eprintln!("Erro ao gerar documentacao: {}", e);
                    std::process::exit(1);
                }
            }
        }

        [_, flag, arquivo] if flag == "--check" || flag == "verificar" => {
            let fonte = match fs::read_to_string(arquivo) {
                Err(e) => {
                    eprintln!("Erro ao abrir '{}': {}", arquivo, e);
                    std::process::exit(1);
                }
                Ok(c) => c,
            };
            let resultado = if e_template(arquivo) {
                template::verificar(&fonte)
            } else {
                verificar_codigo(&fonte)
            };
            match resultado {
                Ok(()) => println!("OK: {}", arquivo),
                Err(e) => {
                    eprintln!("Erro de verificacao em '{}': {}", arquivo, e);
                    std::process::exit(1);
                }
            }
        }

        [_, flag, arquivo] if flag == "--vm" => {
            let fonte = match fs::read_to_string(arquivo) {
                Err(e) => {
                    eprintln!("Erro ao abrir '{}': {}", arquivo, e);
                    std::process::exit(1);
                }
                Ok(c) => c,
            };
            let tokens = match Lexer::novo(&fonte).tokenizar() {
                Ok(t) => t,
                Err(e) => { eprintln!("Erro: {}", e); std::process::exit(1); }
            };
            let ast = match Parser::novo(tokens).parsear() {
                Ok(a) => a,
                Err(e) => { eprintln!("Erro: {}", e); std::process::exit(1); }
            };
            let ops = match compilador::compilar(&ast) {
                Ok(o) => o,
                Err(e) => { eprintln!("Erro de compilacao: {}", e); std::process::exit(1); }
            };
            let mut maquina = vm::Maquina::nova();
            if let Err(e) = maquina.executar(&ops) {
                eprintln!("Erro de execucao (VM): {}", e);
                std::process::exit(1);
            }
        }

        [_, flag, arquivo] if flag == "--vm-reg" => {
            let fonte = fs::read_to_string(arquivo).unwrap_or_else(|e| {
                eprintln!("Erro ao abrir '{}': {}", arquivo, e);
                std::process::exit(1)
            });
            let resultado = (|| -> Result<(), String> {
                let tokens = Lexer::novo(&fonte).tokenizar()?;
                let ast = Parser::novo(tokens).parsear()?;
                let programa = register_vm::compilar(&ast)?;
                register_vm::MaquinaReg::executar(&programa)?;
                Ok(())
            })();
            if let Err(e) = resultado {
                eprintln!("Erro de execucao (VM de registradores): {}", e);
                std::process::exit(1);
            }
        }


        // pep fastcgi <script.pep> [porta]
        [_, cmd, script] if cmd == "fastcgi" => {
            fastcgi::iniciar(script.clone(), 9000);
        }
        [_, cmd, script, porta] if cmd == "fastcgi" => {
            let p: u16 = porta.parse().unwrap_or_else(|_| {
                eprintln!("Porta invalida, usando 9000");
                9000
            });
            fastcgi::iniciar(script.clone(), p);
        }

        // pep servir
        [_, cmd] if cmd == "servir" => {
            servidor::iniciar(7878, std::env::current_dir().unwrap());
        }
        [_, cmd, porta] if cmd == "servir" => {
            let p: u16 = porta.parse().unwrap_or_else(|_| {
                eprintln!("Porta invalida, usando 7878");
                7878
            });
            servidor::iniciar(p, std::env::current_dir().unwrap());
        }
        [_, cmd, porta, dir] if cmd == "servir" => {
            let p: u16 = porta.parse().unwrap_or_else(|_| {
                eprintln!("Porta invalida, usando 7878");
                7878
            });
            servidor::iniciar(p, PathBuf::from(dir));
        }

        // pep arquivo.pep ou arquivo.phtml
        [_, arquivo] => {
            let fonte = match fs::read_to_string(arquivo) {
                Err(e) => {
                    eprintln!("Erro ao abrir '{}': {}", arquivo, e);
                    std::process::exit(1);
                }
                Ok(c) => c,
            };
            let mut interp = Interpretador::novo();
            injetar_contexto_servidor(&mut interp);
            let resultado = if e_template(arquivo) {
                let base = Path::new(arquivo).parent();
                processar_template_com_base(&fonte, &mut interp, base)
            } else {
                let base = Path::new(arquivo).parent();
                executar_codigo_com_base(&fonte, &mut interp, base)
            };
            if let Err(e) = resultado {
                eprintln!("Erro: {}", e);
                std::process::exit(1);
            }
        }

        _ => {
            eprintln!("Uso: pep [--check | --vm | --vm-reg] [arquivo | lsp | formatar arquivo | doc arquivo | servir [porta] [dir]]");
            std::process::exit(1);
        }
    }
}
