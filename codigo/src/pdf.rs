use lopdf::{Document, LoadOptions, PdfMetadata};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone)]
pub struct InformacoesPdf {
    pub titulo: Option<String>,
    pub autor: Option<String>,
    pub assunto: Option<String>,
    pub palavras_chave: Option<String>,
    pub criador: Option<String>,
    pub produtor: Option<String>,
    pub data_criacao: Option<String>,
    pub data_modificacao: Option<String>,
    pub paginas: u32,
    pub versao: String,
}

#[derive(Debug, Clone)]
pub struct OpcoesOcr {
    pub idioma: String,
    pub dpi: u32,
    pub psm: u8,
    pub senha: Option<String>,
}

impl Default for OpcoesOcr {
    fn default() -> Self {
        Self {
            idioma: "por".to_string(),
            dpi: 300,
            psm: 3,
            senha: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DisponibilidadeOcr {
    pub disponivel: bool,
    pub tesseract: bool,
    pub pdftoppm: bool,
}

static PROXIMO_TEMP_OCR: AtomicU64 = AtomicU64::new(1);

struct DiretorioTemporario(PathBuf);

impl Drop for DiretorioTemporario {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn erro(caminho: &str, operacao: &str, erro: impl std::fmt::Display) -> String {
    format!("{}('{}'): {}", operacao, caminho, erro)
}

fn carregar(caminho: &str, senha: Option<&str>) -> Result<Document, String> {
    let opcoes = senha.map(LoadOptions::with_password).unwrap_or_default();
    Document::load_with_options(caminho, opcoes).map_err(|e| erro(caminho, "pdf_abrir", e))
}

fn carregar_metadados(caminho: &str, senha: Option<&str>) -> Result<PdfMetadata, String> {
    match senha {
        Some(senha) => Document::load_metadata_with_password(caminho, senha),
        None => Document::load_metadata(caminho),
    }
    .map_err(|e| erro(caminho, "pdf_informacoes", e))
}

pub fn informacoes(caminho: &str, senha: Option<&str>) -> Result<InformacoesPdf, String> {
    let m = carregar_metadados(caminho, senha)?;
    Ok(InformacoesPdf {
        titulo: m.title,
        autor: m.author,
        assunto: m.subject,
        palavras_chave: m.keywords,
        criador: m.creator,
        produtor: m.producer,
        data_criacao: m.creation_date,
        data_modificacao: m.modification_date,
        paginas: m.page_count,
        versao: m.version,
    })
}

pub fn numero_paginas(caminho: &str, senha: Option<&str>) -> Result<u32, String> {
    Ok(carregar_metadados(caminho, senha)?.page_count)
}

pub fn extrair_paginas(caminho: &str, senha: Option<&str>) -> Result<Vec<String>, String> {
    let documento = carregar(caminho, senha)?;
    documento
        .get_pages()
        .keys()
        .map(|numero| {
            documento
                .extract_text(&[*numero])
                .map_err(|e| format!("pdf_extrair_pagina('{}', {}): {}", caminho, numero, e))
        })
        .collect()
}

pub fn extrair_texto(caminho: &str, senha: Option<&str>) -> Result<String, String> {
    Ok(extrair_paginas(caminho, senha)?.join("\n"))
}

pub fn extrair_pagina(caminho: &str, pagina: u32, senha: Option<&str>) -> Result<String, String> {
    if pagina == 0 {
        return Err("pdf_extrair_pagina: a numeracao comeca em 1".to_string());
    }
    let documento = carregar(caminho, senha)?;
    if !documento.get_pages().contains_key(&pagina) {
        return Err(format!(
            "pdf_extrair_pagina('{}'): pagina {} inexistente",
            caminho, pagina
        ));
    }
    documento
        .extract_text(&[pagina])
        .map_err(|e| format!("pdf_extrair_pagina('{}', {}): {}", caminho, pagina, e))
}

fn executavel(nome_env: &str, padrao: &str) -> String {
    std::env::var(nome_env)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| padrao.to_string())
}

fn comando_disponivel(programa: &str, argumento_versao: &str) -> bool {
    Command::new(programa)
        .arg(argumento_versao)
        .output()
        .map(|saida| saida.status.success())
        .unwrap_or(false)
}

pub fn disponibilidade_ocr() -> DisponibilidadeOcr {
    let tesseract = comando_disponivel(&executavel("PEP_TESSERACT", "tesseract"), "--version");
    let pdftoppm = comando_disponivel(&executavel("PEP_PDFTOPPM", "pdftoppm"), "-v");
    DisponibilidadeOcr {
        disponivel: tesseract && pdftoppm,
        tesseract,
        pdftoppm,
    }
}

fn validar_opcoes_ocr(opcoes: &OpcoesOcr) -> Result<(), String> {
    if opcoes.idioma.trim().is_empty() || opcoes.idioma.len() > 64 {
        return Err("pdf_ocr: idioma invalido".to_string());
    }
    if !(72..=600).contains(&opcoes.dpi) {
        return Err("pdf_ocr: dpi deve estar entre 72 e 600".to_string());
    }
    if opcoes.psm > 13 {
        return Err("pdf_ocr: psm deve estar entre 0 e 13".to_string());
    }
    Ok(())
}

fn novo_diretorio_ocr() -> Result<DiretorioTemporario, String> {
    let id = PROXIMO_TEMP_OCR.fetch_add(1, Ordering::Relaxed);
    let caminho = std::env::temp_dir().join(format!("pep-ocr-{}-{}", std::process::id(), id));
    std::fs::create_dir(&caminho).map_err(|e| {
        format!(
            "pdf_ocr: nao foi possivel criar diretorio temporario: {}",
            e
        )
    })?;
    Ok(DiretorioTemporario(caminho))
}

fn erro_comando(nome: &str, saida: &[u8]) -> String {
    let texto = String::from_utf8_lossy(saida);
    let texto = texto.trim();
    if texto.is_empty() {
        format!("{} terminou com erro", nome)
    } else {
        format!("{}: {}", nome, texto)
    }
}

fn numero_imagem(caminho: &Path) -> u32 {
    caminho
        .file_stem()
        .and_then(|v| v.to_str())
        .and_then(|v| v.rsplit('-').next())
        .and_then(|v| v.parse().ok())
        .unwrap_or(u32::MAX)
}

pub fn ocr_paginas(caminho: &str, opcoes: &OpcoesOcr) -> Result<Vec<String>, String> {
    validar_opcoes_ocr(opcoes)?;
    if !Path::new(caminho).is_file() {
        return Err(format!("pdf_ocr('{}'): arquivo nao encontrado", caminho));
    }

    let tesseract = executavel("PEP_TESSERACT", "tesseract");
    let pdftoppm = executavel("PEP_PDFTOPPM", "pdftoppm");
    let temporario = novo_diretorio_ocr()?;
    let prefixo = temporario.0.join("pagina");

    let mut renderizar = Command::new(&pdftoppm);
    renderizar.arg("-png").arg("-r").arg(opcoes.dpi.to_string());
    if let Some(senha) = &opcoes.senha {
        renderizar.arg("-upw").arg(senha);
    }
    let saida = renderizar.arg(caminho).arg(&prefixo).output().map_err(|e| {
        format!("pdf_ocr: nao foi possivel executar '{}': {}. Instale o Poppler ou defina PEP_PDFTOPPM", pdftoppm, e)
    })?;
    if !saida.status.success() {
        return Err(erro_comando("pdftoppm", &saida.stderr));
    }

    let mut imagens: Vec<PathBuf> = std::fs::read_dir(&temporario.0)
        .map_err(|e| format!("pdf_ocr: erro ao listar paginas renderizadas: {}", e))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")))
        .collect();
    imagens.sort_by_key(|p| numero_imagem(p));
    if imagens.is_empty() {
        return Err("pdf_ocr: o PDF nao produziu paginas para reconhecimento".to_string());
    }

    let mut paginas = Vec::with_capacity(imagens.len());
    for (indice, imagem) in imagens.iter().enumerate() {
        let saida = Command::new(&tesseract)
            .arg(imagem)
            .arg("stdout")
            .arg("-l").arg(&opcoes.idioma)
            .arg("--psm").arg(opcoes.psm.to_string())
            .output()
            .map_err(|e| format!(
                "pdf_ocr: nao foi possivel executar '{}': {}. Instale o Tesseract ou defina PEP_TESSERACT",
                tesseract, e
            ))?;
        if !saida.status.success() {
            return Err(format!(
                "pagina {}: {}",
                indice + 1,
                erro_comando("tesseract", &saida.stderr)
            ));
        }
        paginas.push(String::from_utf8_lossy(&saida.stdout).trim().to_string());
    }
    Ok(paginas)
}

pub fn ocr_texto(caminho: &str, opcoes: &OpcoesOcr) -> Result<String, String> {
    Ok(ocr_paginas(caminho, opcoes)?.join("\n\n"))
}

pub fn extrair_texto_com_ocr(caminho: &str, opcoes: &OpcoesOcr) -> Result<String, String> {
    let senha = opcoes.senha.as_deref();
    let paginas_normais = extrair_paginas(caminho, senha)?;
    if paginas_normais.iter().all(|p| !p.trim().is_empty()) {
        return Ok(paginas_normais.join("\n"));
    }
    let paginas_ocr = ocr_paginas(caminho, opcoes)?;
    Ok(paginas_normais
        .into_iter()
        .enumerate()
        .map(|(i, texto)| {
            if texto.trim().is_empty() {
                paginas_ocr.get(i).cloned().unwrap_or_default()
            } else {
                texto
            }
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::content::{Content, Operation};
    use lopdf::{dictionary, Object, Stream};

    fn criar_pdf(caminho: &std::path::Path) {
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Courier"
        });
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! { "F1" => font_id }
        });
        let content = Content {
            operations: vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 12.into()]),
                Operation::new("Td", vec![50.into(), 700.into()]),
                Operation::new("Tj", vec![Object::string_literal("Texto PDF em PEP")]),
                Operation::new("ET", vec![]),
            ],
        };
        let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page", "Parent" => pages_id, "Contents" => content_id
        });
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages", "Kids" => vec![page_id.into()], "Count" => 1,
                "Resources" => resources_id,
                "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()]
            }),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        let info_id =
            doc.add_object(dictionary! { "Title" => Object::string_literal("Teste PEP") });
        doc.trailer.set("Root", catalog_id);
        doc.trailer.set("Info", info_id);
        doc.save(caminho).unwrap();
    }

    #[test]
    fn le_informacoes_e_extrai_texto() {
        let caminho = std::env::temp_dir().join(format!("pep_pdf_{}.pdf", std::process::id()));
        criar_pdf(&caminho);
        let caminho_texto = caminho.to_string_lossy();

        let info = informacoes(&caminho_texto, None).unwrap();
        assert_eq!(info.paginas, 1);
        assert_eq!(info.titulo.as_deref(), Some("Teste PEP"));
        assert!(extrair_pagina(&caminho_texto, 1, None)
            .unwrap()
            .contains("Texto PDF em PEP"));
        assert!(extrair_texto(&caminho_texto, None)
            .unwrap()
            .contains("Texto PDF em PEP"));
        assert!(extrair_pagina(&caminho_texto, 2, None).is_err());

        let _ = std::fs::remove_file(caminho);
    }

    #[test]
    fn ocr_valida_opcoes_antes_de_executar_ferramentas() {
        let opcoes = OpcoesOcr {
            dpi: 20,
            ..OpcoesOcr::default()
        };
        let erro = ocr_paginas("qualquer.pdf", &opcoes).unwrap_err();
        assert!(erro.contains("dpi deve estar entre 72 e 600"));
    }
}
