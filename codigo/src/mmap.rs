/// Carregamento de modelos/tensores via mapeamento de memória (memmap2).
///
/// Um arquivo mapeado vive no registry global MMAPS até ser fechado.
/// O handle retornado para o PEP é Valor::Inteiro(id).
///
/// Funções expostas ao interpretador:
///   mmap_abrir(caminho)                        → id
///   mmap_tamanho(id)                           → bytes
///   mmap_fechar(id)
///   mmap_ler_f32(id, offset, count)            → Tensor shape=[count]
///   mmap_ler_f64(id, offset, count)            → Tensor shape=[count]
///   mmap_tensor_f32(id, offset, linhas, colunas) → Tensor shape=[linhas, colunas]
///   mmap_tensor_f64(id, offset, linhas, colunas) → Tensor shape=[linhas, colunas]
///   mmap_ler_bytes(id, offset, count)          → Bytes
use std::collections::HashMap;
use std::fs::File;
use std::sync::{OnceLock, RwLock};

use memmap2::Mmap;

use crate::interpretador::Valor;

static MMAPS: OnceLock<RwLock<HashMap<u64, Mmap>>> = OnceLock::new();

fn registry() -> &'static RwLock<HashMap<u64, Mmap>> {
    MMAPS.get_or_init(|| RwLock::new(HashMap::new()))
}

fn proximo_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(1);
    CTR.fetch_add(1, Ordering::Relaxed)
}

pub fn mmap_abrir(caminho: &str) -> Result<u64, String> {
    let arquivo = File::open(caminho)
        .map_err(|e| format!("mmap_abrir: nao foi possivel abrir '{}': {}", caminho, e))?;
    let mapa = unsafe {
        Mmap::map(&arquivo)
            .map_err(|e| format!("mmap_abrir: mmap falhou em '{}': {}", caminho, e))?
    };
    let id = proximo_id();
    registry().write().unwrap().insert(id, mapa);
    Ok(id)
}

pub fn mmap_fechar(id: u64) {
    registry().write().unwrap().remove(&id);
}

pub fn mmap_tamanho(id: u64) -> Result<usize, String> {
    registry()
        .read()
        .unwrap()
        .get(&id)
        .map(|m| m.len())
        .ok_or_else(|| format!("mmap_tamanho: id {} invalido", id))
}

pub fn mmap_ler_f32(id: u64, offset: usize, count: usize) -> Result<Valor, String> {
    let guard = registry().read().unwrap();
    let mapa = guard
        .get(&id)
        .ok_or_else(|| format!("mmap_ler_f32: id {} invalido", id))?;
    let fim = offset + count * 4;
    if fim > mapa.len() {
        return Err(format!(
            "mmap_ler_f32: offset+count ({}) excede tamanho do arquivo ({})",
            fim,
            mapa.len()
        ));
    }
    let dados: Vec<f64> = mapa[offset..fim]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
        .collect();
    Ok(Valor::Tensor {
        shape: vec![count],
        dados: std::sync::Arc::new(dados),
    })
}

pub fn mmap_ler_f64(id: u64, offset: usize, count: usize) -> Result<Valor, String> {
    let guard = registry().read().unwrap();
    let mapa = guard
        .get(&id)
        .ok_or_else(|| format!("mmap_ler_f64: id {} invalido", id))?;
    let fim = offset + count * 8;
    if fim > mapa.len() {
        return Err(format!(
            "mmap_ler_f64: offset+count ({}) excede tamanho do arquivo ({})",
            fim,
            mapa.len()
        ));
    }
    let dados: Vec<f64> = mapa[offset..fim]
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect();
    Ok(Valor::Tensor {
        shape: vec![count],
        dados: std::sync::Arc::new(dados),
    })
}

pub fn mmap_tensor_f32(
    id: u64,
    offset: usize,
    linhas: usize,
    colunas: usize,
) -> Result<Valor, String> {
    let count = linhas * colunas;
    match mmap_ler_f32(id, offset, count)? {
        Valor::Tensor { dados, .. } => Ok(Valor::Tensor {
            shape: vec![linhas, colunas],
            dados,
        }),
        _ => unreachable!(),
    }
}

pub fn mmap_tensor_f64(
    id: u64,
    offset: usize,
    linhas: usize,
    colunas: usize,
) -> Result<Valor, String> {
    let count = linhas * colunas;
    match mmap_ler_f64(id, offset, count)? {
        Valor::Tensor { dados, .. } => Ok(Valor::Tensor {
            shape: vec![linhas, colunas],
            dados,
        }),
        _ => unreachable!(),
    }
}

pub fn mmap_ler_bytes(id: u64, offset: usize, count: usize) -> Result<Valor, String> {
    let guard = registry().read().unwrap();
    let mapa = guard
        .get(&id)
        .ok_or_else(|| format!("mmap_ler_bytes: id {} invalido", id))?;
    let fim = offset + count;
    if fim > mapa.len() {
        return Err(format!(
            "mmap_ler_bytes: offset+count ({}) excede tamanho ({})",
            fim,
            mapa.len()
        ));
    }
    Ok(Valor::Bytes(std::sync::Arc::new(
        mapa[offset..fim].to_vec(),
    )))
}
