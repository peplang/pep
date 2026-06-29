use libloading::{Library, Symbol};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

type FuncaoFfi = unsafe extern "C" fn(*const u8, usize, *mut *mut u8, *mut usize) -> i32;
type LiberarFfi = unsafe extern "C" fn(*mut u8, usize);

const LIMITE_RETORNO: usize = 16 * 1024 * 1024;
static PROXIMO_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static BIBLIOTECAS: RefCell<HashMap<u64, Library>> = RefCell::new(HashMap::new());
}

pub fn permitido() -> bool {
    std::env::var("PEP_FFI_PERMITIR")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

pub fn carregar(caminho: &str) -> Result<u64, String> {
    if !permitido() {
        return Err("FFI desativada; defina PEP_FFI_PERMITIR=1 para habilitar".to_string());
    }
    let path = Path::new(caminho)
        .canonicalize()
        .map_err(|e| format!("ffi_carregar('{}'): {}", caminho, e))?;
    let extensao = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(extensao.as_str(), "dll" | "so" | "dylib") {
        return Err("ffi_carregar: somente .dll, .so e .dylib sao aceitos".to_string());
    }
    let biblioteca = unsafe { Library::new(&path) }
        .map_err(|e| format!("ffi_carregar('{}'): {}", path.display(), e))?;
    let id = PROXIMO_ID.fetch_add(1, Ordering::Relaxed);
    BIBLIOTECAS.with(|b| b.borrow_mut().insert(id, biblioteca));
    Ok(id)
}

pub fn fechar(id: u64) -> bool {
    BIBLIOTECAS.with(|b| b.borrow_mut().remove(&id).is_some())
}

pub fn chamar(id: u64, simbolo: &str, entrada_json: &str) -> Result<String, String> {
    if simbolo.is_empty()
        || !simbolo
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err("ffi_chamar: nome de simbolo invalido".to_string());
    }
    BIBLIOTECAS.with(|bibliotecas| {
        let bibliotecas = bibliotecas.borrow();
        let lib = bibliotecas
            .get(&id)
            .ok_or_else(|| format!("ffi_chamar: biblioteca #{} nao esta carregada", id))?;
        unsafe {
            let funcao: Symbol<FuncaoFfi> = lib
                .get(simbolo.as_bytes())
                .map_err(|e| format!("ffi_chamar: simbolo '{}': {}", simbolo, e))?;
            let liberar: Symbol<LiberarFfi> = lib
                .get(b"pep_ffi_liberar")
                .map_err(|e| format!("ffi_chamar: simbolo obrigatorio 'pep_ffi_liberar': {}", e))?;
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let mut tamanho = 0usize;
            let codigo = funcao(
                entrada_json.as_ptr(),
                entrada_json.len(),
                &mut ptr,
                &mut tamanho,
            );
            if codigo != 0 {
                if !ptr.is_null() {
                    liberar(ptr, tamanho);
                }
                return Err(format!(
                    "ffi_chamar('{}'): plugin retornou codigo {}",
                    simbolo, codigo
                ));
            }
            if ptr.is_null() && tamanho != 0 {
                return Err("ffi_chamar: plugin retornou ponteiro nulo".to_string());
            }
            if tamanho > LIMITE_RETORNO {
                if !ptr.is_null() {
                    liberar(ptr, tamanho);
                }
                return Err(format!(
                    "ffi_chamar: retorno excede {} bytes",
                    LIMITE_RETORNO
                ));
            }
            let bytes = if tamanho == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(ptr, tamanho).to_vec()
            };
            if !ptr.is_null() {
                liberar(ptr, tamanho);
            }
            String::from_utf8(bytes).map_err(|e| format!("ffi_chamar: retorno nao e UTF-8: {}", e))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejeita_simbolo_inseguro() {
        assert!(chamar(999, "funcao;comando", "{}")
            .unwrap_err()
            .contains("simbolo invalido"));
    }
}
