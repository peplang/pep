/// Modo FastCGI
///
/// Protocolo FastCGI 1.0 sobre TCP.
/// Framing: header 8 bytes + body (contentLen + paddingLen bytes)
///   version(1) type(1) requestId(2be) contentLen(2be) paddingLen(1) reserved(1)
///
/// Fluxo por requisição:
///   ← BEGIN_REQUEST (tipo 1)
///   ← PARAMS        (tipo 4, vários; terminado por PARAMS com contentLen=0)
///   ← STDIN         (tipo 5, vários; terminado por STDIN com contentLen=0)
///   → STDOUT        (tipo 6: "Status: 200\r\nContent-Type: ...\r\n\r\n<body>")
///   → STDOUT vazio  (tipo 6, contentLen=0: sinaliza fim)
///   → END_REQUEST   (tipo 3)
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

const FCGI_END_REQUEST: u8 = 3;
const FCGI_PARAMS: u8 = 4;
const FCGI_STDIN: u8 = 5;
const FCGI_STDOUT: u8 = 6;

fn ler_exato(stream: &mut impl Read, n: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

fn ler_record(stream: &mut impl Read) -> io::Result<(u8, u16, Vec<u8>)> {
    let hdr = ler_exato(stream, 8)?;
    let tipo = hdr[1];
    let request_id = u16::from_be_bytes([hdr[2], hdr[3]]);
    let content_len = u16::from_be_bytes([hdr[4], hdr[5]]) as usize;
    let padding_len = hdr[6] as usize;
    let body = ler_exato(stream, content_len)?;
    let _pad = ler_exato(stream, padding_len)?;
    Ok((tipo, request_id, body))
}

fn escrever_record(
    stream: &mut impl Write,
    tipo: u8,
    request_id: u16,
    dados: &[u8],
) -> io::Result<()> {
    for chunk in if dados.is_empty() {
        vec![&dados[..]]
    } else {
        dados.chunks(0xFFFF).collect()
    } {
        let len = chunk.len() as u16;
        let pad = (8 - (chunk.len() % 8)) % 8;
        stream.write_all(&[
            1,
            tipo,
            (request_id >> 8) as u8,
            (request_id & 0xFF) as u8,
            (len >> 8) as u8,
            (len & 0xFF) as u8,
            pad as u8,
            0,
        ])?;
        stream.write_all(chunk)?;
        if pad > 0 {
            stream.write_all(&vec![0u8; pad])?;
        }
    }
    Ok(())
}

/// Decodifica stream de pares name=value no formato FastCGI.
fn decodificar_params(dados: &[u8]) -> Result<HashMap<String, String>, String> {
    let mut mapa = HashMap::new();
    let mut i = 0;
    while i < dados.len() {
        let (nome_len, d1) = fcgi_len(dados, i).ok_or("PARAMS truncado")?;
        i += d1;
        let (val_len, d2) = fcgi_len(dados, i).ok_or("PARAMS truncado")?;
        i += d2;
        if i.saturating_add(nome_len).saturating_add(val_len) > dados.len() {
            return Err("PARAMS com tamanho invalido".to_string());
        }
        let nome = String::from_utf8_lossy(&dados[i..i + nome_len]).to_string();
        i += nome_len;
        let val = String::from_utf8_lossy(&dados[i..i + val_len]).to_string();
        i += val_len;
        mapa.insert(nome, val);
    }
    Ok(mapa)
}

fn fcgi_len(buf: &[u8], pos: usize) -> Option<(usize, usize)> {
    let primeiro = *buf.get(pos)?;
    if primeiro & 0x80 == 0 {
        Some((primeiro as usize, 1))
    } else if pos + 3 < buf.len() {
        let v = ((primeiro as usize & 0x7F) << 24)
            | ((buf[pos + 1] as usize) << 16)
            | ((buf[pos + 2] as usize) << 8)
            | (buf[pos + 3] as usize);
        Some((v, 4))
    } else {
        None
    }
}

fn processar_conexao(mut stream: TcpStream, script: Arc<String>) {
    let debug = std::env::var("PEP_FCGI_DEBUG")
        .ok()
        .is_some_and(|v| v == "1");
    loop {
        // Espera BEGIN_REQUEST
        let (tipo, request_id, begin_body) = match ler_record(&mut stream) {
            Ok(r) => r,
            Err(_) => return,
        };
        if debug {
            eprintln!("FastCGI: record inicial tipo={} id={}", tipo, request_id);
        }
        if tipo != 1 {
            continue;
        } // 1 = BEGIN_REQUEST
        if begin_body.len() < 3 || u16::from_be_bytes([begin_body[0], begin_body[1]]) != 1 {
            let _ = escrever_record(
                &mut stream,
                FCGI_END_REQUEST,
                request_id,
                &[0, 0, 0, 0, 3, 0, 0, 0],
            );
            return;
        }
        let keep_conn = begin_body.get(2).is_some_and(|flags| flags & 1 != 0);

        // Lê PARAMS até record vazio
        let mut params_raw = Vec::new();
        loop {
            match ler_record(&mut stream) {
                Ok((FCGI_PARAMS, id, data)) if id == request_id && data.is_empty() => break,
                Ok((FCGI_PARAMS, id, data)) if id == request_id => {
                    if params_raw.len().saturating_add(data.len()) > 64 * 1024 {
                        return;
                    }
                    params_raw.extend_from_slice(&data)
                }
                Ok((tipo, id, data)) => {
                    if debug {
                        eprintln!(
                            "FastCGI: durante PARAMS tipo={} id={} len={}",
                            tipo,
                            id,
                            data.len()
                        );
                    }
                }
                Err(_) => return,
            }
        }
        let params = match decodificar_params(&params_raw) {
            Ok(p) => p,
            Err(_) => return,
        };
        if debug {
            eprintln!("FastCGI: params completos ({})", params.len());
        }

        // Lê STDIN até record vazio
        let mut stdin_bytes = Vec::new();
        loop {
            match ler_record(&mut stream) {
                Ok((FCGI_STDIN, id, data)) if id == request_id && data.is_empty() => break,
                Ok((FCGI_STDIN, id, data)) if id == request_id => {
                    let limite = std::env::var("PEP_MAX_BODY")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(4 * 1024 * 1024);
                    if stdin_bytes.len().saturating_add(data.len()) > limite {
                        return;
                    }
                    stdin_bytes.extend_from_slice(&data)
                }
                Ok((tipo, id, data)) => {
                    if debug {
                        eprintln!(
                            "FastCGI: durante STDIN tipo={} id={} len={}",
                            tipo,
                            id,
                            data.len()
                        );
                    }
                }
                Err(_) => return,
            }
        }
        if debug {
            eprintln!("FastCGI: stdin completo ({})", stdin_bytes.len());
        }

        // Constrói os campos de requisição a partir das variáveis CGI
        let metodo = params
            .get("REQUEST_METHOD")
            .cloned()
            .unwrap_or_else(|| "GET".to_string());
        let uri = params
            .get("PATH_INFO")
            .filter(|v| !v.is_empty())
            .or_else(|| params.get("REQUEST_URI"))
            .or_else(|| params.get("SCRIPT_NAME"))
            .cloned()
            .unwrap_or_else(|| "/".to_string());
        let caminho = uri
            .split_once('?')
            .map(|(c, _)| c)
            .unwrap_or(&uri)
            .to_string();
        let query_str = params.get("QUERY_STRING").cloned().unwrap_or_else(|| {
            uri.split_once('?')
                .map(|(_, q)| q.to_string())
                .unwrap_or_default()
        });

        let mut cabecalhos: HashMap<String, String> = HashMap::new();
        for (k, v) in &params {
            if let Some(rest) = k.strip_prefix("HTTP_") {
                let nome = rest
                    .split('_')
                    .map(|p| {
                        let mut chars = p.chars();
                        match chars.next() {
                            None => String::new(),
                            Some(f) => {
                                format!("{}{}", f.to_uppercase(), chars.as_str().to_lowercase())
                            }
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("-");
                cabecalhos.insert(nome.to_ascii_lowercase(), v.clone());
            }
        }
        if let Some(ct) = params.get("CONTENT_TYPE") {
            cabecalhos.insert("content-type".to_string(), ct.clone());
        }
        if let Some(cl) = params.get("CONTENT_LENGTH") {
            cabecalhos.insert("content-length".to_string(), cl.clone());
        }

        // Executa o script PEP
        let alvo_requisicao = params
            .get("SCRIPT_FILENAME")
            .map(std::path::Path::new)
            .filter(|p| {
                p.is_dir()
                    || (p.is_file()
                        && matches!(
                            p.extension().and_then(|e| e.to_str()),
                            Some("pep" | "phtml")
                        ))
            })
            .map(|p| p.to_string_lossy().to_string())
            .or_else(|| {
                params
                    .get("DOCUMENT_ROOT")
                    .filter(|p| std::path::Path::new(p).is_dir())
                    .cloned()
            })
            .unwrap_or_else(|| script.as_ref().clone());
        let (status, hdrs, corpo) = crate::servidor::executar_pep_fastcgi(
            &alvo_requisicao,
            metodo,
            caminho,
            query_str,
            cabecalhos,
            stdin_bytes,
        );

        // Monta resposta no formato CGI (Status: + headers + body)
        let mut saida = format!(
            "Status: {} {}\r\n",
            status,
            crate::servidor::texto_status(status)
        );
        for (k, v) in hdrs {
            saida.push_str(&format!("{}: {}\r\n", k, v));
        }
        saida.push_str("\r\n");
        let mut payload = saida.into_bytes();
        payload.extend_from_slice(&corpo);

        // Envia STDOUT + END_REQUEST
        let _ = escrever_record(&mut stream, FCGI_STDOUT, request_id, &payload);
        let _ = escrever_record(&mut stream, FCGI_STDOUT, request_id, &[]);
        // END_REQUEST: appStatus(4be=0) + protocolStatus(1=REQUEST_COMPLETE) + reserved(3)
        let _ = escrever_record(
            &mut stream,
            FCGI_END_REQUEST,
            request_id,
            &[0, 0, 0, 0, 0, 0, 0, 0],
        );
        if !keep_conn {
            return;
        }
    }
}

/// Ponto de entrada: `pep fastcgi <script.pep|diretorio> [porta]`
pub fn iniciar(script: String, porta: u16) {
    let alvo = std::path::Path::new(&script);
    if alvo.is_dir() {
        crate::servidor::carregar_rotas_da_raiz(alvo);
        crate::servidor::carregar_paginas_da_raiz(alvo);
    }
    let ttl = std::env::var("PEP_SESSAO_TTL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    crate::sessoes::inicializar(ttl);
    let addr = format!("127.0.0.1:{}", porta);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("FastCGI: nao foi possivel ouvir em {}: {}", addr, e);
        std::process::exit(1);
    });

    println!("PEP FastCGI ouvindo em {} | alvo: {}", addr, script);
    println!();
    println!("Configuracao Nginx (adicione ao bloco location):");
    println!("  fastcgi_pass {}; ", addr);
    println!("  include fastcgi_params;");
    println!("  fastcgi_param PATH_INFO $uri;");

    let script = Arc::new(script);
    let workers = std::env::var("PEP_FCGI_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .max(2)
        });
    static CONEXOES_ATIVAS: AtomicUsize = AtomicUsize::new(0);
    for conn in listener.incoming() {
        match conn {
            Ok(s) => {
                if std::env::var("PEP_FCGI_DEBUG")
                    .ok()
                    .is_some_and(|v| v == "1")
                {
                    eprintln!("FastCGI: conexao aceita");
                }
                if CONEXOES_ATIVAS.fetch_add(1, Ordering::AcqRel) >= workers {
                    CONEXOES_ATIVAS.fetch_sub(1, Ordering::AcqRel);
                    drop(s);
                    continue;
                }
                let script = Arc::clone(&script);
                thread::spawn(move || {
                    processar_conexao(s, script);
                    CONEXOES_ATIVAS.fetch_sub(1, Ordering::AcqRel);
                });
            }
            Err(e) => eprintln!("FastCGI: erro de aceite: {}", e),
        }
    }
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn rejeita_params_truncados() {
        assert!(decodificar_params(&[0x80]).is_err());
    }

    #[test]
    fn decodifica_params_simples() {
        let dados = [3u8, 3, b'F', b'O', b'O', b'b', b'a', b'r'];
        let params = decodificar_params(&dados).unwrap();
        assert_eq!(params.get("FOO").map(String::as_str), Some("bar"));
    }
}
