use crate::{lexer::Lexer, parser::Parser};
use lsp_server::{Connection, Message, Notification, Request, Response};
use serde_json::{json, Value};
use std::collections::HashMap;

type Resultado<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const PALAVRAS: &[&str] = &[
    "var",
    "se",
    "senao",
    "enquanto",
    "para",
    "funcao",
    "retornar",
    "imprimir",
    "tentar",
    "capturar",
    "finalmente",
    "lancar",
    "importar",
    "incluir",
    "como",
    "verdadeiro",
    "falso",
    "nulo",
    "e",
    "ou",
    "nao",
    "em",
    "de",
    "ate",
    "passo",
];

const NATIVAS: &[&str] = &[
    "texto",
    "numero",
    "inteiro",
    "tipo",
    "tamanho",
    "adicionar",
    "mapa_obter",
    "mapa_definir",
    "ler_arquivo",
    "escrever_arquivo",
    "json_serializar",
    "json_deserializar",
    "sqlite_conectar",
    "sqlite_consultar",
    "sqlite_executar",
    "obter_url",
    "postar_url",
    "pdf_extrair_texto",
    "pdf_extrair_texto_com_ocr",
];

pub fn iniciar_stdio() -> Resultado<()> {
    let (conexao, threads) = Connection::stdio();
    let capacidades = json!({
        "textDocumentSync": 1,
        "completionProvider": { "triggerCharacters": ["."] },
        "hoverProvider": true,
        "documentSymbolProvider": true
    });
    conexao.initialize(capacidades)?;
    let mut documentos: HashMap<String, String> = HashMap::new();

    for mensagem in &conexao.receiver {
        match mensagem {
            Message::Request(requisicao) => {
                if conexao.handle_shutdown(&requisicao)? {
                    break;
                }
                tratar_requisicao(&conexao, requisicao, &documentos)?;
            }
            Message::Notification(notificacao) => {
                tratar_notificacao(&conexao, notificacao, &mut documentos)?;
            }
            Message::Response(_) => {}
        }
    }
    drop(conexao);
    threads.join()?;
    Ok(())
}

fn tratar_notificacao(
    conexao: &Connection,
    notificacao: Notification,
    documentos: &mut HashMap<String, String>,
) -> Resultado<()> {
    match notificacao.method.as_str() {
        "textDocument/didOpen" => {
            let uri = texto_em(&notificacao.params, &["textDocument", "uri"])?;
            let fonte = texto_em(&notificacao.params, &["textDocument", "text"])?;
            documentos.insert(uri.clone(), fonte.clone());
            publicar_diagnosticos(conexao, &uri, &fonte)?;
        }
        "textDocument/didChange" => {
            let uri = texto_em(&notificacao.params, &["textDocument", "uri"])?;
            let fonte = notificacao.params["contentChanges"]
                .as_array()
                .and_then(|v| v.last())
                .and_then(|v| v["text"].as_str())
                .unwrap_or_default()
                .to_string();
            documentos.insert(uri.clone(), fonte.clone());
            publicar_diagnosticos(conexao, &uri, &fonte)?;
        }
        "textDocument/didClose" => {
            let uri = texto_em(&notificacao.params, &["textDocument", "uri"])?;
            documentos.remove(&uri);
            publicar(
                conexao,
                "textDocument/publishDiagnostics",
                json!({"uri": uri, "diagnostics": []}),
            )?;
        }
        _ => {}
    }
    Ok(())
}

fn tratar_requisicao(
    conexao: &Connection,
    req: Request,
    documentos: &HashMap<String, String>,
) -> Resultado<()> {
    let resultado = match req.method.as_str() {
        "textDocument/completion" => Value::Array(PALAVRAS.iter().chain(NATIVAS.iter()).map(|nome| {
            json!({"label": nome, "kind": if NATIVAS.contains(nome) { 3 } else { 14 }})
        }).collect()),
        "textDocument/hover" => hover(&req.params, documentos),
        "textDocument/documentSymbol" => simbolos(&req.params, documentos),
        _ => {
            conexao.sender.send(Message::Response(Response::new_err(
                req.id, -32601, format!("Metodo LSP nao implementado: {}", req.method)
            )))?;
            return Ok(());
        }
    };
    conexao
        .sender
        .send(Message::Response(Response::new_ok(req.id, resultado)))?;
    Ok(())
}

fn publicar_diagnosticos(conexao: &Connection, uri: &str, fonte: &str) -> Resultado<()> {
    let erro = validar(fonte).err();
    let diagnosticos = erro.into_iter().map(|mensagem| {
        let linha = linha_da_mensagem(&mensagem).saturating_sub(1) as u64;
        json!({
            "range": {"start": {"line": linha, "character": 0}, "end": {"line": linha, "character": 1000}},
            "severity": 1, "source": "pep", "message": mensagem
        })
    }).collect::<Vec<_>>();
    publicar(
        conexao,
        "textDocument/publishDiagnostics",
        json!({"uri": uri, "diagnostics": diagnosticos}),
    )
}

fn validar(fonte: &str) -> Result<(), String> {
    let tokens = Lexer::novo(fonte).tokenizar()?;
    Parser::novo(tokens).parsear().map(|_| ())
}

fn hover(params: &Value, documentos: &HashMap<String, String>) -> Value {
    let uri = params["textDocument"]["uri"].as_str().unwrap_or_default();
    let linha = params["position"]["line"].as_u64().unwrap_or(0) as usize;
    let coluna = params["position"]["character"].as_u64().unwrap_or(0) as usize;
    let palavra = documentos
        .get(uri)
        .and_then(|f| palavra_em(f, linha, coluna))
        .unwrap_or_default();
    let descricao = if PALAVRAS.contains(&palavra.as_str()) {
        format!("`{}` — palavra-chave da linguagem PEP", palavra)
    } else if NATIVAS.contains(&palavra.as_str()) {
        format!("`{}(...)` — funcao nativa PEP", palavra)
    } else {
        format!("`{}`", palavra)
    };
    json!({"contents": {"kind": "markdown", "value": descricao}})
}

fn simbolos(params: &Value, documentos: &HashMap<String, String>) -> Value {
    let uri = params["textDocument"]["uri"].as_str().unwrap_or_default();
    let Some(fonte) = documentos.get(uri) else {
        return json!([]);
    };
    let mut saida = Vec::new();
    for (numero, linha) in fonte.lines().enumerate() {
        let limpa = linha.trim_start();
        let (nome, tipo) = if let Some(resto) = limpa.strip_prefix("funcao ") {
            (resto.split('(').next().unwrap_or("").trim(), 12)
        } else if let Some(resto) = limpa.strip_prefix("var ") {
            (
                resto
                    .split(|c| c == '=' || c == ' ')
                    .next()
                    .unwrap_or("")
                    .trim(),
                13,
            )
        } else {
            continue;
        };
        if !nome.is_empty() {
            saida.push(json!({
                "name": nome, "kind": tipo,
                "range": {"start": {"line": numero, "character": 0}, "end": {"line": numero, "character": linha.len()}},
                "selectionRange": {"start": {"line": numero, "character": 0}, "end": {"line": numero, "character": linha.len()}}
            }));
        }
    }
    Value::Array(saida)
}

fn palavra_em(fonte: &str, linha: usize, coluna: usize) -> Option<String> {
    let chars: Vec<char> = fonte.lines().nth(linha)?.chars().collect();
    let pos = coluna.min(chars.len());
    let mut inicio = pos;
    let mut fim = pos;
    while inicio > 0 && (chars[inicio - 1].is_alphanumeric() || chars[inicio - 1] == '_') {
        inicio -= 1;
    }
    while fim < chars.len() && (chars[fim].is_alphanumeric() || chars[fim] == '_') {
        fim += 1;
    }
    Some(chars[inicio..fim].iter().collect())
}

fn linha_da_mensagem(mensagem: &str) -> usize {
    mensagem
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(2)
        .find_map(|p| {
            (p[0].trim_end_matches(':') == "Linha")
                .then(|| p[1].trim_end_matches(':').parse().ok())
                .flatten()
        })
        .unwrap_or(1)
}

fn texto_em(valor: &Value, caminho: &[&str]) -> Resultado<String> {
    let mut atual = valor;
    for chave in caminho {
        atual = &atual[*chave];
    }
    atual
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("campo LSP ausente: {}", caminho.join(".")).into())
}

fn publicar(conexao: &Connection, metodo: &str, params: Value) -> Resultado<()> {
    conexao
        .sender
        .send(Message::Notification(Notification::new(
            metodo.to_string(),
            params,
        )))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encontra_palavra_no_cursor() {
        assert_eq!(
            palavra_em("var resposta = tamanho(lista)", 0, 18).as_deref(),
            Some("tamanho")
        );
    }
    #[test]
    fn valida_codigo_pep() {
        assert!(validar("var x = 1\nimprimir(x)").is_ok());
        assert!(validar("var =").is_err());
    }
}
