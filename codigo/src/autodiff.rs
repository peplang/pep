/// Diferenciação automática reversa (backpropagation) para a VM PEP.
///
/// Design: fita (tape) thread-local. Cada operação `grad_*` chama `registrar_op`,
/// que anota o ID de saída + função backward. `retropropagar` percorre a fita ao
/// contrário e acumula gradientes pelo IDs das entradas.
///
/// Tensores normais (VmValor::Tensor) não participam da fita. Para rastrear
/// gradientes o usuário usa `grad_tensor(t)` → VmValor::TensorGrad.
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// Função backward: dado grad_saida + dados das entradas (por ref) → grads para cada entrada
type GradFn = Box<dyn Fn(&[f64], &[&Vec<f64>]) -> Vec<Vec<f64>>>;

struct Operacao {
    saida_id: usize,
    entradas_ids: Vec<usize>,
    dados_ent: Vec<Arc<Vec<f64>>>,
    backward: GradFn,
}

struct Estado {
    tape: Vec<Operacao>,
    grads: HashMap<usize, Vec<f64>>,
    shapes: HashMap<usize, Vec<usize>>,
    proximo_id: usize,
    gravando: bool,
}

thread_local! {
    static ESTADO: RefCell<Estado> = RefCell::new(Estado {
        tape: Vec::new(),
        grads: HashMap::new(),
        shapes: HashMap::new(),
        proximo_id: 0,
        gravando: false,
    });
}

// ── API pública ───────────────────────────────────────────────────────────────

pub fn ativar() {
    ESTADO.with(|e| e.borrow_mut().gravando = true);
}
pub fn desativar() {
    ESTADO.with(|e| e.borrow_mut().gravando = false);
}

pub fn esta_gravando() -> bool {
    ESTADO.with(|e| e.borrow().gravando)
}

/// Cria um novo nó terminal (folha) — tensor de entrada do grafo.
pub fn novo_tensor(shape: Vec<usize>, dados: Arc<Vec<f64>>) -> usize {
    ESTADO.with(|e| {
        let mut e = e.borrow_mut();
        let id = e.proximo_id;
        e.proximo_id += 1;
        let n: usize = shape.iter().product::<usize>().max(1);
        e.shapes.insert(id, shape);
        e.grads.insert(id, vec![0.0; n]);
        let _ = dados; // dados de folha não precisam ser guardadas (sem backward)
        id
    })
}

pub fn obter_grad(id: usize) -> Option<(Vec<usize>, Vec<f64>)> {
    ESTADO.with(|e| {
        let e = e.borrow();
        let grad = e.grads.get(&id)?.clone();
        let shape = e.shapes.get(&id)?.clone();
        Some((shape, grad))
    })
}

pub fn zerar_grad(id: usize) {
    ESTADO.with(|e| {
        if let Some(g) = e.borrow_mut().grads.get_mut(&id) {
            g.iter_mut().for_each(|x| *x = 0.0);
        }
    });
}

pub fn zerar_todos() {
    ESTADO.with(|e| {
        for g in e.borrow_mut().grads.values_mut() {
            g.iter_mut().for_each(|x| *x = 0.0);
        }
    });
}

pub fn limpar() {
    ESTADO.with(|e| {
        let mut e = e.borrow_mut();
        e.tape.clear();
        e.grads.clear();
        e.shapes.clear();
        e.proximo_id = 0;
        e.gravando = false;
    });
}

/// Backpropagation a partir de `id_loss`. Inicializa grad da perda com 1s,
/// percorre a fita ao contrário e acumula gradientes.
pub fn retropropagar(id_loss: usize) -> Result<(), String> {
    ESTADO.with(|e| {
        let mut e = e.borrow_mut();
        // Inicializa gradiente da perda com 1.0
        match e.grads.get_mut(&id_loss) {
            Some(g) => g.iter_mut().for_each(|x| *x = 1.0),
            None => {
                return Err(format!(
                    "grad_retropropagar: ID {} não encontrado na fita",
                    id_loss
                ))
            }
        }
        // Percorre a fita ao contrário
        for i in (0..e.tape.len()).rev() {
            let saida_id = e.tape[i].saida_id;
            let grad_saida: Vec<f64> = e.grads.get(&saida_id).cloned().unwrap_or_default();
            let dados_refs: Vec<Arc<Vec<f64>>> = e.tape[i].dados_ent.clone();
            let refs: Vec<&Vec<f64>> = dados_refs.iter().map(|d| d.as_ref()).collect();
            let grads_ent = (e.tape[i].backward)(&grad_saida, &refs);
            let entradas_ids = e.tape[i].entradas_ids.clone();
            for (k, &id) in entradas_ids.iter().enumerate() {
                if k < grads_ent.len() {
                    if let Some(g) = e.grads.get_mut(&id) {
                        for (a, b) in g.iter_mut().zip(grads_ent[k].iter()) {
                            *a += b;
                        }
                    }
                }
            }
        }
        Ok(())
    })
}

// ── Utilitários internos ──────────────────────────────────────────────────────

fn registrar_op(
    entradas_ids: Vec<usize>,
    dados_ent: Vec<Arc<Vec<f64>>>,
    shape_saida: Vec<usize>,
    backward: GradFn,
) -> usize {
    ESTADO.with(|e| {
        let mut e = e.borrow_mut();
        let id = e.proximo_id;
        e.proximo_id += 1;
        let n: usize = shape_saida.iter().product::<usize>().max(1);
        e.shapes.insert(id, shape_saida);
        e.grads.insert(id, vec![0.0; n]);
        e.tape.push(Operacao {
            saida_id: id,
            entradas_ids,
            dados_ent,
            backward,
        });
        id
    })
}

fn matmul_grad(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn transpor_grad(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

// ── Operações com rastreamento de gradiente ───────────────────────────────────

pub fn op_soma(
    sa: Vec<usize>,
    da: Arc<Vec<f64>>,
    ia: usize,
    sb: Vec<usize>,
    db: Arc<Vec<f64>>,
    ib: usize,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    if sa != sb {
        return Err(format!(
            "grad_soma: shapes incompatíveis {:?} vs {:?}",
            sa, sb
        ));
    }
    let out = Arc::new(
        da.iter()
            .zip(db.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<_>>(),
    );
    if !esta_gravando() {
        return Ok((sa.clone(), out, novo_tensor(sa.clone(), da)));
    }
    let id = registrar_op(
        vec![ia, ib],
        vec![da, db],
        sa.clone(),
        Box::new(move |g, _| vec![g.to_vec(), g.to_vec()]),
    );
    Ok((sa, out, id))
}

pub fn op_sub(
    sa: Vec<usize>,
    da: Arc<Vec<f64>>,
    ia: usize,
    sb: Vec<usize>,
    db: Arc<Vec<f64>>,
    ib: usize,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    if sa != sb {
        return Err(format!(
            "grad_sub: shapes incompatíveis {:?} vs {:?}",
            sa, sb
        ));
    }
    let out = Arc::new(
        da.iter()
            .zip(db.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<_>>(),
    );
    if !esta_gravando() {
        return Ok((sa.clone(), out, novo_tensor(sa.clone(), da)));
    }
    let id = registrar_op(
        vec![ia, ib],
        vec![da, db],
        sa.clone(),
        Box::new(move |g, _| vec![g.to_vec(), g.iter().map(|x| -x).collect()]),
    );
    Ok((sa, out, id))
}

pub fn op_mul(
    sa: Vec<usize>,
    da: Arc<Vec<f64>>,
    ia: usize,
    sb: Vec<usize>,
    db: Arc<Vec<f64>>,
    ib: usize,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    if sa != sb {
        return Err(format!(
            "grad_mul: shapes incompatíveis {:?} vs {:?}",
            sa, sb
        ));
    }
    let out = Arc::new(
        da.iter()
            .zip(db.iter())
            .map(|(x, y)| x * y)
            .collect::<Vec<_>>(),
    );
    if !esta_gravando() {
        return Ok((sa.clone(), out, novo_tensor(sa.clone(), da)));
    }
    // d(a*b)/da = b, d(a*b)/db = a
    let id = registrar_op(
        vec![ia, ib],
        vec![da, db],
        sa.clone(),
        Box::new(move |g, dados| {
            let b = &dados[1];
            let a = &dados[0];
            let ga: Vec<f64> = g.iter().zip(b.iter()).map(|(gi, bi)| gi * bi).collect();
            let gb: Vec<f64> = g.iter().zip(a.iter()).map(|(gi, ai)| gi * ai).collect();
            vec![ga, gb]
        }),
    );
    Ok((sa, out, id))
}

pub fn op_matmul(
    sa: Vec<usize>,
    da: Arc<Vec<f64>>,
    ia: usize,
    sb: Vec<usize>,
    db: Arc<Vec<f64>>,
    ib: usize,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    if sa.len() != 2 || sb.len() != 2 {
        return Err(format!(
            "grad_matmul: requer tensores 2D, recebeu {:?} e {:?}",
            sa, sb
        ));
    }
    let (m, k, n) = (sa[0], sa[1], sb[1]);
    if k != sb[0] {
        return Err(format!(
            "grad_matmul: dims incompatíveis {}x{} @ {}x{}",
            m, k, sb[0], n
        ));
    }
    let out = Arc::new(matmul_grad(&da, m, k, &db, n));
    if !esta_gravando() {
        return Ok((vec![m, n], out, novo_tensor(vec![m, n], da)));
    }
    let shape_out = vec![m, n];
    let id = registrar_op(
        vec![ia, ib],
        vec![da, db],
        shape_out.clone(),
        Box::new(move |g, dados| {
            let a = &dados[0];
            let b = &dados[1];
            // grad_A = grad_out @ B^T  → (m,n) @ (n,k) = (m,k)
            let bt = transpor_grad(b, k, n);
            let ga = matmul_grad(g, m, n, &bt, k);
            // grad_B = A^T @ grad_out  → (k,m) @ (m,n) = (k,n)
            let at = transpor_grad(a, m, k);
            let gb = matmul_grad(&at, k, m, g, n);
            vec![ga, gb]
        }),
    );
    Ok((shape_out, out, id))
}

pub fn op_relu(
    s: Vec<usize>,
    d: Arc<Vec<f64>>,
    i: usize,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    let out = Arc::new(d.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>());
    if !esta_gravando() {
        return Ok((s.clone(), out, novo_tensor(s.clone(), d)));
    }
    let id = registrar_op(
        vec![i],
        vec![d],
        s.clone(),
        Box::new(move |g, dados| {
            let x = &dados[0];
            vec![g
                .iter()
                .zip(x.iter())
                .map(|(gi, &xi)| if xi > 0.0 { *gi } else { 0.0 })
                .collect()]
        }),
    );
    Ok((s, out, id))
}

pub fn op_sigmoid(
    s: Vec<usize>,
    d: Arc<Vec<f64>>,
    i: usize,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    let sig: Vec<f64> = d.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    let out = Arc::new(sig);
    let out_clone = out.clone();
    if !esta_gravando() {
        return Ok((s.clone(), out, novo_tensor(s.clone(), d)));
    }
    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    let id = registrar_op(
        vec![i],
        vec![d],
        s.clone(),
        Box::new(move |g, _| {
            let ga: Vec<f64> = g
                .iter()
                .zip(out_clone.iter())
                .map(|(gi, &sx)| gi * sx * (1.0 - sx))
                .collect();
            vec![ga]
        }),
    );
    Ok((s, out, id))
}

pub fn op_tanh(
    s: Vec<usize>,
    d: Arc<Vec<f64>>,
    i: usize,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    let th: Vec<f64> = d.iter().map(|&x| x.tanh()).collect();
    let out = Arc::new(th);
    let out_clone = out.clone();
    if !esta_gravando() {
        return Ok((s.clone(), out, novo_tensor(s.clone(), d)));
    }
    let id = registrar_op(
        vec![i],
        vec![d],
        s.clone(),
        Box::new(move |g, _| {
            // d/dx tanh(x) = 1 - tanh(x)^2
            let ga: Vec<f64> = g
                .iter()
                .zip(out_clone.iter())
                .map(|(gi, &tx)| gi * (1.0 - tx * tx))
                .collect();
            vec![ga]
        }),
    );
    Ok((s, out, id))
}

pub fn op_exp(
    s: Vec<usize>,
    d: Arc<Vec<f64>>,
    i: usize,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    let ex: Vec<f64> = d.iter().map(|&x| x.exp()).collect();
    let out = Arc::new(ex);
    let out_clone = out.clone();
    if !esta_gravando() {
        return Ok((s.clone(), out, novo_tensor(s.clone(), d)));
    }
    let id = registrar_op(
        vec![i],
        vec![d],
        s.clone(),
        Box::new(move |g, _| {
            let ga: Vec<f64> = g
                .iter()
                .zip(out_clone.iter())
                .map(|(gi, &ex)| gi * ex)
                .collect();
            vec![ga]
        }),
    );
    Ok((s, out, id))
}

pub fn op_log(
    s: Vec<usize>,
    d: Arc<Vec<f64>>,
    i: usize,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    let out = Arc::new(d.iter().map(|&x| x.ln()).collect::<Vec<_>>());
    if !esta_gravando() {
        return Ok((s.clone(), out, novo_tensor(s.clone(), d)));
    }
    let id = registrar_op(
        vec![i],
        vec![d],
        s.clone(),
        Box::new(move |g, dados| {
            let ga: Vec<f64> = g
                .iter()
                .zip(dados[0].iter())
                .map(|(gi, &xi)| gi / xi)
                .collect();
            vec![ga]
        }),
    );
    Ok((s, out, id))
}

pub fn op_neg(
    s: Vec<usize>,
    d: Arc<Vec<f64>>,
    i: usize,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    let out = Arc::new(d.iter().map(|&x| -x).collect::<Vec<_>>());
    if !esta_gravando() {
        return Ok((s.clone(), out, novo_tensor(s.clone(), d)));
    }
    let id = registrar_op(
        vec![i],
        vec![d],
        s.clone(),
        Box::new(move |g, _| vec![g.iter().map(|x| -x).collect()]),
    );
    Ok((s, out, id))
}

pub fn op_soma_total(
    s: Vec<usize>,
    d: Arc<Vec<f64>>,
    i: usize,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    let total: f64 = d.iter().sum();
    let out = Arc::new(vec![total]);
    if !esta_gravando() {
        return Ok((vec![1], out, novo_tensor(vec![1], d)));
    }
    let n = d.len();
    let id = registrar_op(
        vec![i],
        vec![d],
        s.clone(),
        Box::new(move |g, _| {
            // d(soma)/dx_i = 1 para todo i; grad de entrada = g[0] repetido n vezes
            let grad_escalar = if g.is_empty() { 1.0 } else { g[0] };
            vec![vec![grad_escalar; n]]
        }),
    );
    Ok((s, out, id))
}

pub fn op_escalar_mul(
    s: Vec<usize>,
    d: Arc<Vec<f64>>,
    i: usize,
    escalar: f64,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    let out = Arc::new(d.iter().map(|&x| x * escalar).collect::<Vec<_>>());
    if !esta_gravando() {
        return Ok((s.clone(), out, novo_tensor(s.clone(), d)));
    }
    let id = registrar_op(
        vec![i],
        vec![d],
        s.clone(),
        Box::new(move |g, _| vec![g.iter().map(|x| x * escalar).collect()]),
    );
    Ok((s, out, id))
}
