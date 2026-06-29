/// Repositorio de sessoes HTTP in-memory, thread-safe.
///
/// Substitui o antigo sistema de arquivos JSON (pep_sessoes/).
/// Uma unica instancia global e compartilhada entre todos os workers via Arc<RwLock<>>.
/// Limpeza de sessoes expiradas e feita periodicamente em background.
use crate::interpretador::Valor;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

// -- Tipos --------------------------------------------------------------------

struct Sessao {
    dados: HashMap<String, Valor>,
    expira_em: Instant,
    ttl: Duration,
}

pub struct Repositorio {
    sessoes: HashMap<String, Sessao>,
    pub ttl_padrao: Duration,
}

impl Repositorio {
    pub fn novo(ttl_minutos: u64) -> Self {
        Repositorio {
            sessoes: HashMap::new(),
            ttl_padrao: Duration::from_secs(ttl_minutos * 60),
        }
    }

    fn valida(&self, id: &str) -> bool {
        self.sessoes
            .get(id)
            .map_or(false, |s| s.expira_em > Instant::now())
    }

    pub fn garantir(&mut self, id: &str) {
        if !self.valida(id) {
            let ttl = self.ttl_padrao;
            self.sessoes.insert(
                id.to_string(),
                Sessao {
                    dados: HashMap::new(),
                    expira_em: Instant::now() + ttl,
                    ttl,
                },
            );
        }
    }

    pub fn obter(&self, id: &str, chave: &str) -> Option<Valor> {
        self.sessoes
            .get(id)
            .filter(|s| s.expira_em > Instant::now())
            .and_then(|s| s.dados.get(chave))
            .cloned()
    }

    pub fn definir(&mut self, id: &str, chave: String, valor: Valor) {
        self.garantir(id);
        if let Some(s) = self.sessoes.get_mut(id) {
            s.dados.insert(chave, valor);
        }
    }

    pub fn remover(&mut self, id: &str, chave: &str) -> Option<Valor> {
        self.sessoes.get_mut(id)?.dados.remove(chave)
    }

    pub fn destruir(&mut self, id: &str) {
        self.sessoes.remove(id);
    }

    pub fn migrar(&mut self, id_antigo: &str, id_novo: &str) {
        if let Some(sessao) = self.sessoes.remove(id_antigo) {
            self.sessoes.insert(id_novo.to_string(), sessao);
        } else {
            self.garantir(id_novo);
        }
    }

    pub fn renovar(&mut self, id: &str) {
        if let Some(s) = self.sessoes.get_mut(id) {
            s.expira_em = Instant::now() + s.ttl;
        }
    }

    pub fn definir_ttl(&mut self, id: &str, minutos: u64) {
        self.garantir(id);
        if let Some(s) = self.sessoes.get_mut(id) {
            s.ttl = Duration::from_secs(minutos * 60);
            s.expira_em = Instant::now() + s.ttl;
        }
    }

    pub fn listar_chaves(&self, id: &str) -> Vec<String> {
        self.sessoes
            .get(id)
            .filter(|s| s.expira_em > Instant::now())
            .map(|s| s.dados.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub fn obter_todos(&self, id: &str) -> HashMap<String, Valor> {
        self.sessoes
            .get(id)
            .filter(|s| s.expira_em > Instant::now())
            .map(|s| s.dados.clone())
            .unwrap_or_default()
    }

    pub fn existe(&self, id: &str) -> bool {
        self.valida(id)
    }

    pub fn limpar_expiradas(&mut self) {
        let agora = Instant::now();
        self.sessoes.retain(|_, s| s.expira_em > agora);
    }

    pub fn total_ativas(&self) -> usize {
        let agora = Instant::now();
        self.sessoes
            .values()
            .filter(|s| s.expira_em > agora)
            .count()
    }
}

// -- Singleton global ---------------------------------------------------------

static REPOSITORIO: OnceLock<Arc<RwLock<Repositorio>>> = OnceLock::new();

/// Deve ser chamado UMA vez em `servidor::iniciar()` antes de aceitar conexoes.
pub fn inicializar(ttl_minutos: u64) {
    let repo = Arc::new(RwLock::new(Repositorio::novo(ttl_minutos)));
    let _ = REPOSITORIO.set(repo.clone());

    // Thread de limpeza: remove sessoes expiradas a cada 60 segundos.
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(60));
        if let Ok(mut r) = repo.write() {
            r.limpar_expiradas();
        }
    });
}

pub fn repo() -> Arc<RwLock<Repositorio>> {
    // Se ainda nao foi inicializado (modo REPL/script), cria um repositorio vazio.
    REPOSITORIO
        .get_or_init(|| Arc::new(RwLock::new(Repositorio::novo(30))))
        .clone()
}

// -- Thread-local: ID da sessao corrente do request ---------------------------

use std::cell::RefCell;

thread_local! {
    static SESSAO_ID: RefCell<String> = RefCell::new(String::new());
}

pub fn definir_sessao_atual(id: String) {
    SESSAO_ID.with(|s| *s.borrow_mut() = id);
}

pub fn obter_sessao_atual() -> String {
    SESSAO_ID.with(|s| s.borrow().clone())
}
