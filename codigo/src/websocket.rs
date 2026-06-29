/// Suporte a WebSocket para o servidor PEP.
///
/// Modelo: uma thread por conexão (blocking I/O).
/// Cada chamada a `ws_aceitar()` faz o HTTP Upgrade e bloqueia a thread no loop do handler.
///
/// API exposta ao PEP:
///   ws_aceitar()            → Nulo   — faz upgrade HTTP→WS
///   ws_receber()            → Texto | Bytes | Nulo (conexão fechada)
///   ws_enviar(msg)          → Nulo
///   ws_enviar_bytes(bytes)  → Nulo
///   ws_fechar()             → Nulo
///   ws_id()                 → Inteiro (ID desta conexão)
///   ws_conexoes()           → Lista[Inteiro]
///   ws_enviar_para(id, msg) → Nulo
///   ws_broadcast(msg)       → Nulo
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use sha1::{Digest, Sha1};

// ── Tipos ──────────────────────────────────────────────────────────────────

pub enum WsMensagem {
    Texto(String),
    Binario(Vec<u8>),
    Fechado,
}

struct WsConn {
    id: u64,
    stream: Arc<Mutex<TcpStream>>,
}

// ── Thread-locals ─────────────────────────────────────────────────────────
// Stream HTTP bruto, definido por servidor.rs antes de invocar o handler.

thread_local! {
    static STREAM_HTTP: RefCell<Option<(TcpStream, String)>> = RefCell::new(None);
    static CONN: RefCell<Option<WsConn>> = RefCell::new(None);
}

// ── Registry global (broadcast / ws_enviar_para) ──────────────────────────

static REGISTRY: OnceLock<Mutex<HashMap<u64, Arc<Mutex<TcpStream>>>>> = OnceLock::new();
static CTR: AtomicU64 = AtomicU64::new(1);

fn registry() -> &'static Mutex<HashMap<u64, Arc<Mutex<TcpStream>>>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── Interface com servidor.rs ─────────────────────────────────────────────

/// Guarda o stream HTTP e a chave WS antes de chamar o handler.
pub fn definir_stream_http(stream: TcpStream, ws_key: String) {
    STREAM_HTTP.with(|s| *s.borrow_mut() = Some((stream, ws_key)));
}

/// Remove estado WS da thread; chamado após o handler retornar.
pub fn limpar_thread() {
    STREAM_HTTP.with(|s| *s.borrow_mut() = None);
    CONN.with(|c| {
        if let Some(conn) = c.borrow_mut().take() {
            if let Ok(mut reg) = registry().lock() {
                reg.remove(&conn.id);
            }
        }
    });
}

// ── Handshake (RFC 6455) ──────────────────────────────────────────────────

pub fn ws_aceitar() -> Result<u64, String> {
    let (mut stream, ws_key) = STREAM_HTTP
        .with(|s| s.borrow_mut().take())
        .ok_or("ws_aceitar: nao ha conexao HTTP pendente para upgrade (chamou ws_aceitar() fora de uma rota WebSocket?)")?;

    let mut hasher = Sha1::new();
    hasher.update(ws_key.trim().as_bytes());
    hasher.update(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    let accept = B64.encode(hasher.finalize());

    let resposta = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {}\r\n\r\n",
        accept
    );
    stream
        .write_all(resposta.as_bytes())
        .map_err(|e| format!("ws_aceitar: falha ao enviar 101: {}", e))?;

    let id = CTR.fetch_add(1, Ordering::Relaxed);
    let arc = Arc::new(Mutex::new(stream));
    registry().lock().unwrap().insert(id, arc.clone());
    CONN.with(|c| *c.borrow_mut() = Some(WsConn { id, stream: arc }));
    Ok(id)
}

// ── Framing ───────────────────────────────────────────────────────────────

fn escrever_frame(stream: &mut TcpStream, opcode: u8, dados: &[u8]) -> std::io::Result<()> {
    let len = dados.len();
    stream.write_all(&[0x80 | opcode])?; // FIN=1, RSV=0
    if len < 126 {
        stream.write_all(&[len as u8])?;
    } else if len < 65536 {
        stream.write_all(&[126u8])?;
        stream.write_all(&(len as u16).to_be_bytes())?;
    } else {
        stream.write_all(&[127u8])?;
        stream.write_all(&(len as u64).to_be_bytes())?;
    }
    if !dados.is_empty() {
        stream.write_all(dados)?;
    }
    Ok(())
}

fn ler_frame(stream: &mut TcpStream) -> Result<WsMensagem, String> {
    loop {
        let mut hdr = [0u8; 2];
        if stream.read_exact(&mut hdr).is_err() {
            return Ok(WsMensagem::Fechado);
        }
        let opcode = hdr[0] & 0x0F;
        let masked = (hdr[1] & 0x80) != 0;
        let mut payload_len = (hdr[1] & 0x7F) as usize;

        if payload_len == 126 {
            let mut ext = [0u8; 2];
            if stream.read_exact(&mut ext).is_err() {
                return Ok(WsMensagem::Fechado);
            }
            payload_len = u16::from_be_bytes(ext) as usize;
        } else if payload_len == 127 {
            let mut ext = [0u8; 8];
            if stream.read_exact(&mut ext).is_err() {
                return Ok(WsMensagem::Fechado);
            }
            payload_len = u64::from_be_bytes(ext) as usize;
        }

        let mask = if masked {
            let mut k = [0u8; 4];
            if stream.read_exact(&mut k).is_err() {
                return Ok(WsMensagem::Fechado);
            }
            Some(k)
        } else {
            None
        };

        let mut dados = vec![0u8; payload_len];
        if payload_len > 0 && stream.read_exact(&mut dados).is_err() {
            return Ok(WsMensagem::Fechado);
        }
        if let Some(key) = mask {
            for (i, b) in dados.iter_mut().enumerate() {
                *b ^= key[i % 4];
            }
        }

        match opcode {
            0x1 => {
                return Ok(WsMensagem::Texto(
                    String::from_utf8_lossy(&dados).to_string(),
                ))
            }
            0x2 => return Ok(WsMensagem::Binario(dados)),
            0x8 => {
                let _ = escrever_frame(stream, 0x8, &[]);
                return Ok(WsMensagem::Fechado);
            }
            0x9 => {
                let _ = escrever_frame(stream, 0xA, &dados);
            } // Ping → Pong
            _ => {}
        }
    }
}

// ── API pública ───────────────────────────────────────────────────────────

pub fn ws_receber() -> Result<Option<WsMensagem>, String> {
    CONN.with(|c| {
        let borrow = c.borrow();
        match borrow.as_ref() {
            None => Ok(None),
            Some(conn) => {
                let mut stream = conn.stream.lock().unwrap();
                match ler_frame(&mut *stream) {
                    Ok(WsMensagem::Fechado) => Ok(None),
                    Ok(msg) => Ok(Some(msg)),
                    Err(_) => Ok(None),
                }
            }
        }
    })
}

pub fn ws_enviar(msg: &str) -> Result<(), String> {
    CONN.with(|c| {
        let borrow = c.borrow();
        match borrow.as_ref() {
            None => Err("ws_enviar: nenhuma conexao WebSocket ativa".to_string()),
            Some(conn) => {
                let mut s = conn.stream.lock().unwrap();
                escrever_frame(&mut *s, 0x1, msg.as_bytes())
                    .map_err(|e| format!("ws_enviar: {}", e))
            }
        }
    })
}

pub fn ws_enviar_bytes(dados: &[u8]) -> Result<(), String> {
    CONN.with(|c| {
        let borrow = c.borrow();
        match borrow.as_ref() {
            None => Err("ws_enviar_bytes: nenhuma conexao WebSocket ativa".to_string()),
            Some(conn) => {
                let mut s = conn.stream.lock().unwrap();
                escrever_frame(&mut *s, 0x2, dados).map_err(|e| format!("ws_enviar_bytes: {}", e))
            }
        }
    })
}

pub fn ws_fechar() {
    CONN.with(|c| {
        if let Some(conn) = c.borrow().as_ref() {
            if let Ok(mut s) = conn.stream.lock() {
                let _ = escrever_frame(&mut *s, 0x8, &[]);
            }
            registry().lock().unwrap().remove(&conn.id);
        }
        *c.borrow_mut() = None;
    });
}

pub fn ws_id() -> Option<u64> {
    CONN.with(|c| c.borrow().as_ref().map(|conn| conn.id))
}

pub fn ws_conexoes() -> Vec<u64> {
    registry().lock().unwrap().keys().cloned().collect()
}

pub fn ws_enviar_para(id: u64, msg: &str) -> Result<(), String> {
    let guard = registry().lock().unwrap();
    match guard.get(&id) {
        None => Err(format!(
            "ws_enviar_para: conexao {} nao encontrada (pode ja ter fechado)",
            id
        )),
        Some(arc) => {
            if let Ok(mut s) = arc.lock() {
                escrever_frame(&mut *s, 0x1, msg.as_bytes())
                    .map_err(|e| format!("ws_enviar_para({}): {}", id, e))
            } else {
                Err(format!(
                    "ws_enviar_para: nao foi possivel bloquear stream {}",
                    id
                ))
            }
        }
    }
}

pub fn ws_broadcast(msg: &str) {
    let guard = registry().lock().unwrap();
    for arc in guard.values() {
        if let Ok(mut s) = arc.lock() {
            let _ = escrever_frame(&mut *s, 0x1, msg.as_bytes());
        }
    }
}
