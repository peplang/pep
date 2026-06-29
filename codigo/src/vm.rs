/// VM de bytecode PEP — máquina de pilha com Arc<str> para chaves (clone O(1))
use crate::bytecode::Op;
use crate::interpretador::Interpretador;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ── Dispositivo de execução tensor ───────────────────────────────────────────

/// Indica onde os dados do tensor residem fisicamente.
/// CPU é o único back-end disponível por padrão.
/// CUDA e Metal ficam disponíveis ao compilar com `--features gpu-cuda` / `--features gpu-metal`.
#[derive(Debug, Clone, PartialEq)]
pub enum Dispositivo {
    Cpu,
    /// Índice da GPU NVIDIA (0 = primeira)
    Cuda(u32),
    /// Apple Metal GPU
    Metal,
}

impl std::fmt::Display for Dispositivo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dispositivo::Cpu => write!(f, "cpu"),
            Dispositivo::Cuda(n) => write!(f, "cuda:{}", n),
            Dispositivo::Metal => write!(f, "metal"),
        }
    }
}

// Registro thread-local: ponteiro Arc (endereço dos dados) → Dispositivo
// Mantém o dispositivo de cada tensor sem alterar o enum VmValor.
use std::cell::RefCell;
thread_local! {
    static DISPOSITIVOS: RefCell<HashMap<usize, Dispositivo>> = RefCell::new(HashMap::new());
}

fn ptr_tensor(dados: &Arc<Vec<f64>>) -> usize {
    Arc::as_ptr(dados) as usize
}

fn registrar_dispositivo(dados: &Arc<Vec<f64>>, d: Dispositivo) {
    DISPOSITIVOS.with(|m| {
        m.borrow_mut().insert(ptr_tensor(dados), d);
    });
}

fn obter_dispositivo(dados: &Arc<Vec<f64>>) -> Dispositivo {
    DISPOSITIVOS.with(|m| {
        m.borrow()
            .get(&ptr_tensor(dados))
            .cloned()
            .unwrap_or(Dispositivo::Cpu)
    })
}

// ── Valor da VM ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum VmValor {
    Int(i64),
    Num(f64),
    Str(String),
    Bool(bool),
    Null,
    Lista(Vec<VmValor>),
    Mapa(HashMap<String, VmValor>),
    Funcao {
        params: Vec<Arc<str>>,
        corpo: Vec<Op>,
    },
    Tensor {
        shape: Vec<usize>,
        dados: Arc<Vec<f64>>,
    },
    /// Tensor com rastreamento de gradiente (autodiff reverso). `id` indexa na
    /// fita thread-local em crate::autodiff. Ao passar por ops normais é tratado
    /// como Tensor (sem registrar na fita); use grad_* para rastrear gradientes.
    TensorGrad {
        shape: Vec<usize>,
        dados: Arc<Vec<f64>>,
        id: usize,
    },
    Bytes(Arc<Vec<u8>>),
    ConexaoBD(u64),
    ConexaoSQLite(u64),
    Erro {
        tipo: String,
        mensagem: String,
        pilha: Vec<String>,
    },
}

impl fmt::Display for VmValor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmValor::Int(n) => write!(f, "{}", n),
            VmValor::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            VmValor::Str(s) => write!(f, "{}", s),
            VmValor::Bool(b) => write!(f, "{}", if *b { "verdadeiro" } else { "falso" }),
            VmValor::Null => write!(f, "nulo"),
            VmValor::Lista(v) => {
                let s: Vec<String> = v.iter().map(|x| x.repr()).collect();
                write!(f, "[{}]", s.join(", "))
            }
            VmValor::Mapa(m) => {
                let mut ps: Vec<String> = m
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.repr()))
                    .collect();
                ps.sort();
                write!(f, "{{{}}}", ps.join(", "))
            }
            VmValor::Funcao { .. } => write!(f, "<função>"),
            VmValor::Tensor { shape, dados } => {
                let mut off = 0usize;
                write!(f, "{}", vm_tensor_fmt(shape, dados, 0, &mut off))
            }
            VmValor::TensorGrad { shape, dados, id } => {
                let mut off = 0usize;
                write!(
                    f,
                    "<grad#{} {}>",
                    id,
                    vm_tensor_fmt(shape, dados, 0, &mut off)
                )
            }
            VmValor::Bytes(b) => write!(f, "<bytes:{}>", b.len()),
            VmValor::ConexaoBD(id) => write!(f, "<conexao BD #{}>", id),
            VmValor::ConexaoSQLite(id) => write!(f, "<conexao SQLite #{}>", id),
            VmValor::Erro { tipo, mensagem, .. } => write!(f, "[Erro:{}] {}", tipo, mensagem),
        }
    }
}

impl VmValor {
    fn repr(&self) -> String {
        match self {
            VmValor::Str(s) => format!("\"{}\"", s),
            o => o.to_string(),
        }
    }
    fn e_verdadeiro(&self) -> bool {
        match self {
            VmValor::Bool(b) => *b,
            VmValor::Null => false,
            VmValor::Num(n) => *n != 0.0,
            VmValor::Int(n) => *n != 0,
            VmValor::Str(s) => !s.is_empty(),
            VmValor::Lista(l) => !l.is_empty(),
            VmValor::Mapa(m) => !m.is_empty(),
            VmValor::Funcao { .. } => true,
            VmValor::Tensor { .. } => true,
            VmValor::TensorGrad { .. } => true,
            VmValor::Bytes(b) => !b.is_empty(),
            VmValor::ConexaoBD(_) | VmValor::ConexaoSQLite(_) => true,
            VmValor::Erro { .. } => false,
        }
    }
}

impl PartialEq for VmValor {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (VmValor::Num(a), VmValor::Num(b)) => a == b,
            (VmValor::Int(a), VmValor::Int(b)) => a == b,
            (VmValor::Int(a), VmValor::Num(b)) | (VmValor::Num(b), VmValor::Int(a)) => {
                *a as f64 == *b
            }
            (VmValor::Str(a), VmValor::Str(b)) => a == b,
            (VmValor::Bool(a), VmValor::Bool(b)) => a == b,
            (VmValor::Null, VmValor::Null) => true,
            (VmValor::Lista(a), VmValor::Lista(b)) => a == b,
            (VmValor::Mapa(a), VmValor::Mapa(b)) => a == b,
            (VmValor::Bytes(a), VmValor::Bytes(b)) => a == b,
            (
                VmValor::Tensor {
                    shape: sa,
                    dados: da,
                },
                VmValor::Tensor {
                    shape: sb,
                    dados: db,
                },
            ) => sa == sb && da == db,
            (VmValor::TensorGrad { id: a, .. }, VmValor::TensorGrad { id: b, .. }) => a == b,
            _ => false,
        }
    }
}

fn vm_tensor_fmt(shape: &[usize], dados: &[f64], dim: usize, off: &mut usize) -> String {
    if dim == shape.len() - 1 {
        let mut s = String::from("[");
        for i in 0..shape[dim] {
            if i > 0 {
                s.push_str(", ");
            }
            let v = dados[*off];
            *off += 1;
            if v.fract() == 0.0 && v.abs() < 1e15 {
                s.push_str(&format!("{}", v as i64));
            } else {
                s.push_str(&format!("{}", v));
            }
        }
        s.push(']');
        s
    } else {
        let mut s = String::from("[");
        for i in 0..shape[dim] {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&vm_tensor_fmt(shape, dados, dim + 1, off));
        }
        s.push(']');
        s
    }
}

// ── Frame de execução ─────────────────────────────────────────────────────────

struct Frame {
    ops: Vec<Op>,
    ip: usize,
    pilha: Vec<VmValor>,
    vars: HashMap<Arc<str>, VmValor>,
    catch_stack: Vec<usize>,
    contadores: Vec<(f64, f64, f64)>,
}

impl Frame {
    fn novo(ops: Vec<Op>, vars: HashMap<Arc<str>, VmValor>) -> Self {
        Frame {
            ops,
            ip: 0,
            pilha: Vec::new(),
            vars,
            catch_stack: Vec::new(),
            contadores: Vec::new(),
        }
    }
    fn push(&mut self, v: VmValor) {
        self.pilha.push(v);
    }
    fn pop(&mut self) -> Result<VmValor, String> {
        self.pilha
            .pop()
            .ok_or_else(|| "Pilha vazia (bug interno da VM)".to_string())
    }
    fn peek(&self) -> Result<&VmValor, String> {
        self.pilha
            .last()
            .ok_or_else(|| "Pilha vazia (bug interno da VM)".to_string())
    }
}

// ── Máquina Virtual ───────────────────────────────────────────────────────────

pub struct Maquina {
    globais: HashMap<Arc<str>, VmValor>,
    nativas: Interpretador,
    diretorios: Vec<PathBuf>,
    incluidos: std::collections::HashSet<PathBuf>,
    operacoes: u64,
    limite_operacoes: u64,
}

impl Maquina {
    pub fn nova() -> Self {
        let limite_operacoes = std::env::var("PEP_MAX_OPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000_000);
        Maquina {
            globais: HashMap::new(),
            nativas: Interpretador::novo(),
            diretorios: Vec::new(),
            incluidos: std::collections::HashSet::new(),
            operacoes: 0,
            limite_operacoes,
        }
    }

    pub fn com_base(base: PathBuf) -> Self {
        let mut vm = Self::nova();
        vm.diretorios.push(base);
        vm
    }

    pub fn definir_global(&mut self, nome: impl Into<Arc<str>>, valor: VmValor) {
        self.globais.insert(nome.into(), valor);
    }

    pub fn definir_globais(&mut self, valores: HashMap<Arc<str>, VmValor>) {
        self.globais.extend(valores);
    }

    pub fn globais(&self) -> HashMap<Arc<str>, VmValor> {
        self.globais.clone()
    }

    pub fn executar(&mut self, ops: &[Op]) -> Result<(), String> {
        self.executar_com_retorno(ops).map(|_| ())
    }

    pub fn executar_com_retorno(&mut self, ops: &[Op]) -> Result<VmValor, String> {
        let mut frame = Frame::novo(ops.to_vec(), self.globais.clone());
        let resultado = self.executar_frame(&mut frame);
        self.globais = frame.vars;
        resultado.map(|v| v.unwrap_or(VmValor::Null))
    }

    pub fn chamar_funcao(
        &mut self,
        funcao: VmValor,
        args: Vec<VmValor>,
    ) -> Result<VmValor, String> {
        match funcao {
            VmValor::Funcao { params, corpo } => {
                if params.len() != args.len() {
                    return Err(format!(
                        "Funcao espera {} arg(s), recebeu {}",
                        params.len(),
                        args.len()
                    ));
                }
                let mut vars = self.globais.clone();
                for (p, a) in params.into_iter().zip(args) {
                    vars.insert(p, a);
                }
                let mut frame = Frame::novo(corpo, vars);
                self.executar_frame(&mut frame)
                    .map(|v| v.unwrap_or(VmValor::Null))
            }
            outro => Err(format!("{} nao e uma funcao", outro)),
        }
    }

    fn executar_frame(&mut self, frame: &mut Frame) -> Result<Option<VmValor>, String> {
        loop {
            if frame.ip >= frame.ops.len() {
                break;
            }
            self.operacoes = self.operacoes.saturating_add(1);
            if self.limite_operacoes > 0 && self.operacoes > self.limite_operacoes {
                return Err(format!(
                    "Limite de {} operacoes excedido na VM",
                    self.limite_operacoes
                ));
            }
            let op = frame.ops[frame.ip].clone();
            frame.ip += 1;

            match op {
                Op::Halt => break,

                Op::PushNum(n) => frame.push(VmValor::Num(n)),
                Op::PushInt(n) => frame.push(VmValor::Int(n)),
                Op::PushStr(s) => frame.push(VmValor::Str(s)),
                Op::PushBool(b) => frame.push(VmValor::Bool(b)),
                Op::PushNull => frame.push(VmValor::Null),

                Op::Load(nome) => {
                    let v = frame
                        .vars
                        .get(&nome)
                        .or_else(|| self.globais.get(&nome))
                        .cloned()
                        .ok_or_else(|| format!("Variável '{}' não definida", nome))?;
                    frame.push(v);
                }
                Op::Store(nome) => {
                    let v = frame.pop()?;
                    frame.vars.insert(nome, v);
                }

                Op::MakeList(n) => {
                    let mut lista = vec![VmValor::Null; n];
                    for i in (0..n).rev() {
                        lista[i] = frame.pop()?;
                    }
                    frame.push(VmValor::Lista(lista));
                }
                Op::MakeMap(chaves) => {
                    let n = chaves.len();
                    let mut valores = vec![VmValor::Null; n];
                    for i in (0..n).rev() {
                        valores[i] = frame.pop()?;
                    }
                    let mapa: HashMap<String, VmValor> = chaves
                        .into_iter()
                        .map(|k| k.to_string())
                        .zip(valores)
                        .collect();
                    frame.push(VmValor::Mapa(mapa));
                }
                Op::GetIndex => {
                    let idx = frame.pop()?;
                    let obj = frame.pop()?;
                    frame.push(vm_get_index(obj, idx)?);
                }
                Op::SetIndex => {
                    let val = frame.pop()?;
                    let idx = frame.pop()?;
                    let obj = frame.pop()?;
                    frame.push(vm_set_index(obj, idx, val)?);
                }

                Op::Add => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_add(a, b)?);
                }
                Op::Sub => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_num(a, b, "-", |x, y| x - y)?);
                }
                Op::Mul => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_mul(a, b)?);
                }
                Op::Div => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_num(a, b, "/", |x, y| {
                        if y == 0.0 {
                            f64::INFINITY
                        } else {
                            x / y
                        }
                    })?);
                }
                Op::IntDiv => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    let divisor = vm_para_f64(&b)?;
                    if divisor == 0.0 {
                        return Err("Divisao inteira por zero".to_string());
                    }
                    frame.push(VmValor::Int((vm_para_f64(&a)? / divisor).trunc() as i64));
                }
                Op::Mod => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_num(a, b, "%", |x, y| x % y)?);
                }
                Op::Neg => match frame.pop()? {
                    VmValor::Int(n) => frame.push(VmValor::Int(-n)),
                    VmValor::Num(n) => frame.push(VmValor::Num(-n)),
                    v => return Err(format!("'-' requer número, recebeu {}", v)),
                },
                Op::Not => {
                    let a = frame.pop()?;
                    frame.push(VmValor::Bool(!a.e_verdadeiro()));
                }

                Op::Eq => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(VmValor::Bool(a == b));
                }
                Op::Ne => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(VmValor::Bool(a != b));
                }
                Op::Lt => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_cmp(a, b, "<")?);
                }
                Op::Gt => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_cmp(a, b, ">")?);
                }
                Op::Le => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_cmp(a, b, "<=")?);
                }
                Op::Ge => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_cmp(a, b, ">=")?);
                }
                Op::And => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(VmValor::Bool(a.e_verdadeiro() && b.e_verdadeiro()));
                }
                Op::Or => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(VmValor::Bool(a.e_verdadeiro() || b.e_verdadeiro()));
                }

                Op::Jump(alvo) => {
                    frame.ip = alvo;
                }
                Op::JumpFalse(alvo) => {
                    let v = frame.pop()?;
                    if !v.e_verdadeiro() {
                        frame.ip = alvo;
                    }
                }
                Op::JumpTrue(alvo) => {
                    let v = frame.pop()?;
                    if v.e_verdadeiro() {
                        frame.ip = alvo;
                    }
                }

                Op::DefFunc {
                    nome: _,
                    params,
                    corpo,
                } => {
                    frame.push(VmValor::Funcao { params, corpo });
                }

                Op::Call(n_args) => {
                    let mut args: Vec<VmValor> =
                        (0..n_args).map(|_| frame.pop()).collect::<Result<_, _>>()?;
                    args.reverse();
                    match frame.pop()? {
                        VmValor::Funcao { params, corpo } => {
                            if params.len() != args.len() {
                                return Err(format!(
                                    "Função espera {} arg(s), recebeu {}",
                                    params.len(),
                                    args.len()
                                ));
                            }
                            let mut vars: HashMap<Arc<str>, VmValor> = HashMap::new();
                            for (k, v) in &frame.vars {
                                vars.insert(k.clone(), v.clone());
                            }
                            for (p, a) in params.iter().zip(args) {
                                vars.insert(p.clone(), a);
                            }
                            let mut sub = Frame::novo(corpo, vars);
                            let ret = self.executar_frame(&mut sub)?;
                            frame.push(ret.unwrap_or(VmValor::Null));
                        }
                        v => return Err(format!("{} não é uma função", v)),
                    }
                }

                Op::CallNative(nome, n_args) => {
                    let mut args: Vec<VmValor> =
                        (0..n_args).map(|_| frame.pop()).collect::<Result<_, _>>()?;
                    args.reverse();

                    let user_func = frame
                        .vars
                        .get(&nome)
                        .or_else(|| self.globais.get(&nome))
                        .cloned();
                    match user_func {
                        Some(VmValor::Funcao { params, corpo }) => {
                            if params.len() != args.len() {
                                return Err(format!(
                                    "Função '{}' espera {} arg(s), recebeu {}",
                                    nome,
                                    params.len(),
                                    args.len()
                                ));
                            }
                            let mut vars: HashMap<Arc<str>, VmValor> = HashMap::new();
                            for (k, v) in &frame.vars {
                                vars.insert(k.clone(), v.clone());
                            }
                            for (p, a) in params.iter().zip(args) {
                                vars.insert(p.clone(), a);
                            }
                            let mut sub = Frame::novo(corpo, vars);
                            let ret = self.executar_frame(&mut sub)?;
                            frame.push(ret.unwrap_or(VmValor::Null));
                        }
                        _ => {
                            let resultado = self.chamar_nativa(&nome, args, &mut frame.vars)?;
                            frame.push(resultado);
                        }
                    }
                }

                Op::Return => {
                    let v = frame.pop()?;
                    return Ok(Some(v));
                }
                Op::ReturnNull => return Ok(Some(VmValor::Null)),

                Op::Include {
                    caminho,
                    obrigatorio,
                } => {
                    self.executar_include(&caminho, obrigatorio, frame)?;
                }
                Op::Import { caminho, alias } => {
                    self.executar_import(&caminho, alias.as_deref(), frame)?;
                }

                Op::Print(n) => {
                    let partes: Vec<String> = (0..n)
                        .map(|_| frame.pop())
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .rev()
                        .map(|v| v.to_string())
                        .collect();
                    crate::interpretador::saida_vm_escrever(&format!("{}\n", partes.join(" ")));
                    frame.push(VmValor::Null);
                }
                Op::Write(n) => {
                    let partes: Vec<String> = (0..n)
                        .map(|_| frame.pop())
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .rev()
                        .map(|v| v.to_string())
                        .collect();
                    crate::interpretador::saida_vm_escrever(&partes.join(""));
                    crate::interpretador::saida_vm_flush();
                    frame.push(VmValor::Null);
                }

                Op::Pop => {
                    frame.pop()?;
                }
                Op::Dup => {
                    let v = frame.peek()?.clone();
                    frame.push(v);
                }

                Op::IterStart { var } => {
                    let passo = match frame.pop()? {
                        VmValor::Num(n) => n,
                        v => return Err(format!("IterStart passo: {}", v)),
                    };
                    let fim = match frame.pop()? {
                        VmValor::Num(n) => n,
                        v => return Err(format!("IterStart fim: {}", v)),
                    };
                    let ini = match frame.pop()? {
                        VmValor::Num(n) => n,
                        v => return Err(format!("IterStart ini: {}", v)),
                    };
                    let in_range = (ini - fim) * passo <= 0.0;
                    frame.contadores.push((ini, fim, passo));
                    if in_range {
                        frame.vars.insert(var, VmValor::Num(ini));
                    }
                    frame.push(VmValor::Bool(in_range));
                }
                Op::IterNext { var, loop_ini } => {
                    if let Some(cnt) = frame.contadores.last_mut() {
                        cnt.0 += cnt.2;
                        let (cur, fim, passo) = *cnt;
                        if (cur - fim) * passo <= 0.0 {
                            frame.vars.insert(var, VmValor::Num(cur));
                            frame.ip = loop_ini;
                        } else {
                            frame.contadores.pop();
                        }
                    } else {
                        return Err("IterNext: sem contador ativo".to_string());
                    }
                }

                Op::Em => {
                    let colecao = frame.pop()?;
                    let valor = frame.pop()?;
                    frame.push(VmValor::Bool(vm_em(&valor, &colecao)?));
                }
                Op::NaoEm => {
                    let colecao = frame.pop()?;
                    let valor = frame.pop()?;
                    frame.push(VmValor::Bool(!vm_em(&valor, &colecao)?));
                }

                Op::TryCatch(offset) => {
                    frame.catch_stack.push(offset);
                }
                Op::EndTry => {
                    frame.catch_stack.pop();
                }
                Op::Throw => {
                    let v = frame.pop()?;
                    let msg = v.to_string();
                    if let Some(catch_ip) = frame.catch_stack.pop() {
                        frame.ip = catch_ip;
                        frame.push(v);
                    } else {
                        return Err(msg);
                    }
                }

                // ── Opcodes tensoriais dedicados ─────────────────────────────
                Op::TensorMatMul => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_tensor_matmul(a, b)?);
                }
                Op::TensorTranspor => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_transpor(t)?);
                }
                // Ativações
                Op::TensorReLU => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_map(t, |x| x.max(0.0), "tensor_relu")?);
                }
                Op::TensorSigmoid => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_map(
                        t,
                        |x| 1.0 / (1.0 + (-x).exp()),
                        "tensor_sigmoid",
                    )?);
                }
                Op::TensorSoftmax => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_softmax(t)?);
                }
                Op::TensorTanh => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_map(t, |x| x.tanh(), "tensor_tanh")?);
                }
                // Element-wise com dois operandos
                Op::TensorAdd => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_tensor_binop(a, b, |x, y| x + y, "tensor_add")?);
                }
                Op::TensorSub => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_tensor_binop(a, b, |x, y| x - y, "tensor_sub")?);
                }
                Op::TensorMulElem => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_tensor_binop(a, b, |x, y| x * y, "tensor_mul")?);
                }
                Op::TensorDivElem => {
                    let b = frame.pop()?;
                    let a = frame.pop()?;
                    frame.push(vm_tensor_binop(a, b, |x, y| x / y, "tensor_div")?);
                }
                Op::TensorPow => {
                    let exp = vm_para_f64(&frame.pop()?).map_err(|e| {
                        format!("tensor_potencia: expoente deve ser número — {}", e)
                    })?;
                    let a = frame.pop()?;
                    frame.push(vm_tensor_map(a, |x| x.powf(exp), "tensor_potencia")?);
                }
                // Unárias
                Op::TensorNeg => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_map(t, |x| -x, "tensor_neg")?);
                }
                Op::TensorExp => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_map(t, |x| x.exp(), "tensor_exp")?);
                }
                Op::TensorLog => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_map(t, |x| x.ln(), "tensor_log")?);
                }
                Op::TensorSqrt => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_map(t, |x| x.sqrt(), "tensor_sqrt")?);
                }
                // Reduções globais
                Op::TensorSomaTotal => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_reducao_global(
                        t,
                        |acc, x| acc + x,
                        0.0,
                        "tensor_soma_total",
                    )?);
                }
                Op::TensorMediaTotal => {
                    let t = frame.pop()?;
                    match t {
                        VmValor::Tensor { ref dados, .. } if dados.is_empty() => {
                            return Err("tensor_media: tensor vazio".to_string())
                        }
                        VmValor::Tensor { ref dados, .. } => {
                            let n = dados.len() as f64;
                            let s: f64 = dados.iter().sum();
                            frame.push(VmValor::Num(s / n));
                        }
                        v => return Err(format!("tensor_media: requer tensor, recebeu {}", v)),
                    }
                }
                Op::TensorMaxTotal => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_reducao_global(
                        t,
                        f64::max,
                        f64::NEG_INFINITY,
                        "tensor_max",
                    )?);
                }
                Op::TensorMinTotal => {
                    let t = frame.pop()?;
                    frame.push(vm_tensor_reducao_global(
                        t,
                        f64::min,
                        f64::INFINITY,
                        "tensor_min",
                    )?);
                }
                // Reduções por eixo
                Op::TensorSomaEixo => {
                    let eixo = vm_para_usize(&frame.pop()?, "tensor_soma_eixo")?;
                    let t = frame.pop()?;
                    frame.push(vm_tensor_reducao_eixo(
                        t,
                        eixo,
                        |acc, x| acc + x,
                        0.0,
                        false,
                        "tensor_soma_eixo",
                    )?);
                }
                Op::TensorMediaEixo => {
                    let eixo = vm_para_usize(&frame.pop()?, "tensor_media_eixo")?;
                    let t = frame.pop()?;
                    frame.push(vm_tensor_reducao_eixo(
                        t,
                        eixo,
                        |acc, x| acc + x,
                        0.0,
                        true,
                        "tensor_media_eixo",
                    )?);
                }
                Op::TensorMaxEixo => {
                    let eixo = vm_para_usize(&frame.pop()?, "tensor_max_eixo")?;
                    let t = frame.pop()?;
                    frame.push(vm_tensor_reducao_eixo(
                        t,
                        eixo,
                        f64::max,
                        f64::NEG_INFINITY,
                        false,
                        "tensor_max_eixo",
                    )?);
                }
                Op::TensorMinEixo => {
                    let eixo = vm_para_usize(&frame.pop()?, "tensor_min_eixo")?;
                    let t = frame.pop()?;
                    frame.push(vm_tensor_reducao_eixo(
                        t,
                        eixo,
                        f64::min,
                        f64::INFINITY,
                        false,
                        "tensor_min_eixo",
                    )?);
                }
                // Forma
                Op::TensorConcatenar => {
                    let eixo = vm_para_usize(&frame.pop()?, "tensor_concatenar")?;
                    let lista = frame.pop()?;
                    frame.push(vm_tensor_concatenar(lista, eixo)?);
                }
                Op::TensorEmpilhar => {
                    let eixo = vm_para_usize(&frame.pop()?, "tensor_empilhar")?;
                    let lista = frame.pop()?;
                    frame.push(vm_tensor_empilhar(lista, eixo)?);
                }
            }
        }
        Ok(None)
    }

    fn resolver_caminho(&self, caminho: &str) -> PathBuf {
        let p = PathBuf::from(caminho);
        if p.is_absolute() {
            return p;
        }
        for dir in self.diretorios.iter().rev() {
            let candidato = dir.join("pep_modules").join(caminho);
            if candidato.exists() {
                return candidato;
            }
            let candidato = dir.join(caminho);
            if candidato.exists() {
                return candidato;
            }
        }
        let cwd = std::env::current_dir().unwrap_or_default();
        let candidato = cwd.join("pep_modules").join(caminho);
        if candidato.exists() {
            return candidato;
        }
        PathBuf::from(caminho)
    }

    fn executar_include(
        &mut self,
        caminho: &str,
        obrigatorio: bool,
        frame: &mut Frame,
    ) -> Result<(), String> {
        let path = self.resolver_caminho(caminho);
        if !path.exists() {
            return if obrigatorio {
                Err(format!("requerer '{}': arquivo não encontrado", caminho))
            } else {
                Ok(())
            };
        }
        let abs =
            std::fs::canonicalize(&path).map_err(|e| format!("incluir '{}': {}", caminho, e))?;
        if !self.incluidos.insert(abs.clone()) {
            return Ok(());
        }
        let fonte = std::fs::read_to_string(&abs)
            .map_err(|e| format!("incluir '{}': {}", abs.display(), e))?;
        let ops = compilar_fonte_para_ops(&abs, &fonte)?;
        if let Some(dir) = abs.parent() {
            self.diretorios.push(dir.to_path_buf());
        }
        let mut sub = Frame::novo(ops, frame.vars.clone());
        let resultado = self.executar_frame(&mut sub);
        if abs.parent().is_some() {
            self.diretorios.pop();
        }
        resultado?;
        frame.vars.extend(sub.vars);
        Ok(())
    }

    fn executar_import(
        &mut self,
        caminho: &str,
        alias: Option<&str>,
        frame: &mut Frame,
    ) -> Result<(), String> {
        let path = self.resolver_caminho(caminho);
        let abs =
            std::fs::canonicalize(&path).map_err(|e| format!("importar '{}': {}", caminho, e))?;
        let fonte = std::fs::read_to_string(&abs)
            .map_err(|e| format!("importar '{}': {}", abs.display(), e))?;
        let ops = compilar_fonte_para_ops(&abs, &fonte)?;
        if let Some(dir) = abs.parent() {
            self.diretorios.push(dir.to_path_buf());
        }
        let mut sub = Frame::novo(ops, HashMap::new());
        let resultado = self.executar_frame(&mut sub);
        if abs.parent().is_some() {
            self.diretorios.pop();
        }
        resultado?;
        if let Some(prefixo) = alias {
            let modulo: HashMap<String, VmValor> = sub
                .vars
                .iter()
                .filter(|(k, _)| !k.starts_with('_'))
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            frame.vars.insert(Arc::from(prefixo), VmValor::Mapa(modulo));
        } else {
            for (k, v) in sub.vars {
                if !k.starts_with('_') {
                    frame.vars.insert(k, v);
                }
            }
        }
        Ok(())
    }

    fn chamar_nativa(
        &self,
        nome: &str,
        mut args: Vec<VmValor>,
        vars: &mut HashMap<Arc<str>, VmValor>,
    ) -> Result<VmValor, String> {
        match nome {
            "escrever" => {
                use std::io::Write as _;
                for v in &args {
                    print!("{}", v);
                }
                let _ = std::io::stdout().flush();
                Ok(VmValor::Null)
            }
            "imprimir" => {
                let ps: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                println!("{}", ps.join(" "));
                Ok(VmValor::Null)
            }
            "nova_linha" => {
                println!();
                Ok(VmValor::Null)
            }
            "texto" => Ok(VmValor::Str(vm_arg1(&args, "texto")?.to_string())),
            "numero" => match vm_arg1(&args, "numero")? {
                VmValor::Num(n) => Ok(VmValor::Num(*n)),
                VmValor::Int(n) => Ok(VmValor::Num(*n as f64)),
                VmValor::Str(s) => s
                    .trim()
                    .parse::<f64>()
                    .map(VmValor::Num)
                    .map_err(|_| format!("Não é possível converter '{}' para número", s)),
                VmValor::Bool(b) => Ok(VmValor::Num(if *b { 1.0 } else { 0.0 })),
                v => Err(format!("Não é possível converter '{}' para número", v)),
            },
            "inteiro" => match vm_arg1(&args, "inteiro")? {
                VmValor::Int(n) => Ok(VmValor::Int(*n)),
                VmValor::Num(n) => Ok(VmValor::Int(n.trunc() as i64)),
                VmValor::Bool(b) => Ok(VmValor::Int(if *b { 1 } else { 0 })),
                VmValor::Str(s) => s
                    .trim()
                    .parse::<i64>()
                    .map(VmValor::Int)
                    .or_else(|_| {
                        s.trim()
                            .parse::<f64>()
                            .map(|f| VmValor::Int(f.trunc() as i64))
                    })
                    .map_err(|_| format!("'inteiro' não pode converter '{}'", s)),
                v => Err(format!("'inteiro' requer número, recebeu {}", v)),
            },
            "tipo" => {
                let t = match vm_arg1(&args, "tipo")? {
                    VmValor::Num(_) => "numero",
                    VmValor::Int(_) => "inteiro",
                    VmValor::Str(_) => "texto",
                    VmValor::Bool(_) => "booleano",
                    VmValor::Null => "nulo",
                    VmValor::Lista(_) => "lista",
                    VmValor::Mapa(_) => "mapa",
                    VmValor::Funcao { .. } => "funcao",
                    VmValor::Tensor { .. } => "tensor",
                    VmValor::TensorGrad { .. } => "tensor_grad",
                    VmValor::Bytes(_) => "bytes",
                    VmValor::ConexaoBD(_) => "conexao_bd",
                    VmValor::ConexaoSQLite(_) => "conexao_sqlite",
                    VmValor::Erro { .. } => "erro",
                };
                Ok(VmValor::Str(t.to_string()))
            }
            "tamanho" => match vm_arg1(&args, "tamanho")? {
                VmValor::Lista(v) => Ok(VmValor::Num(v.len() as f64)),
                VmValor::Str(s) => Ok(VmValor::Num(s.chars().count() as f64)),
                VmValor::Mapa(m) => Ok(VmValor::Num(m.len() as f64)),
                v => Err(format!("'tamanho' requer lista/texto/mapa, recebeu {}", v)),
            },
            "adicionar" => {
                if args.len() != 2 {
                    return Err("'adicionar' requer 2 argumentos".to_string());
                }
                match &args[0] {
                    VmValor::Lista(v) => {
                        let mut lista = v.clone();
                        lista.push(args[1].clone());
                        Ok(VmValor::Lista(lista))
                    }
                    v => Err(format!("'adicionar' requer lista, recebeu {}", v)),
                }
            }
            "intervalo" => match args.as_slice() {
                [VmValor::Num(fim)] => Ok(VmValor::Lista(
                    (0..*fim as i64).map(|i| VmValor::Num(i as f64)).collect(),
                )),
                [VmValor::Num(ini), VmValor::Num(fim)] => Ok(VmValor::Lista(
                    (*ini as i64..*fim as i64)
                        .map(|i| VmValor::Num(i as f64))
                        .collect(),
                )),
                _ => Err("intervalo(fim) ou intervalo(ini, fim)".to_string()),
            },
            "maiusculas" => match vm_arg1(&args, "maiusculas")? {
                VmValor::Str(s) => Ok(VmValor::Str(s.to_uppercase())),
                v => Err(format!("'maiusculas' requer texto, recebeu {}", v)),
            },
            "minusculas" => match vm_arg1(&args, "minusculas")? {
                VmValor::Str(s) => Ok(VmValor::Str(s.to_lowercase())),
                v => Err(format!("'minusculas' requer texto, recebeu {}", v)),
            },
            "aparar" => match vm_arg1(&args, "aparar")? {
                VmValor::Str(s) => Ok(VmValor::Str(s.trim().to_string())),
                v => Err(format!("'aparar' requer texto, recebeu {}", v)),
            },
            "raiz" => match vm_arg1(&args, "raiz")? {
                VmValor::Num(n) => Ok(VmValor::Num(n.sqrt())),
                v => Err(format!("'raiz' requer número, recebeu {}", v)),
            },
            "absoluto" => match vm_arg1(&args, "absoluto")? {
                VmValor::Num(n) => Ok(VmValor::Num(n.abs())),
                v => Err(format!("'absoluto' requer número, recebeu {}", v)),
            },
            "arredondar" => match args.as_slice() {
                [VmValor::Num(n)] => Ok(VmValor::Num(n.round())),
                [VmValor::Num(n), VmValor::Num(c)] => {
                    let f = 10f64.powi(*c as i32);
                    Ok(VmValor::Num((n * f).round() / f))
                }
                _ => Err("arredondar(n) ou arredondar(n, casas)".to_string()),
            },
            "piso" => match vm_arg1(&args, "piso")? {
                VmValor::Num(n) => Ok(VmValor::Num(n.floor())),
                v => Err(format!("'piso' requer número, recebeu {}", v)),
            },
            "teto" => match vm_arg1(&args, "teto")? {
                VmValor::Num(n) => Ok(VmValor::Num(n.ceil())),
                v => Err(format!("'teto' requer número, recebeu {}", v)),
            },
            "pi" => Ok(VmValor::Num(std::f64::consts::PI)),
            "minimo" => match args.as_slice() {
                [VmValor::Num(a), VmValor::Num(b)] => Ok(VmValor::Num(a.min(*b))),
                _ => Err("minimo(a,b)".to_string()),
            },
            "maximo" => match args.as_slice() {
                [VmValor::Num(a), VmValor::Num(b)] => Ok(VmValor::Num(a.max(*b))),
                _ => Err("maximo(a,b)".to_string()),
            },
            "mapa" => Ok(VmValor::Mapa(HashMap::new())),
            "mapa_definir" => match args.as_slice() {
                [VmValor::Mapa(m), VmValor::Str(k), v] => {
                    let mut m2 = m.clone();
                    m2.insert(k.clone(), v.clone());
                    Ok(VmValor::Mapa(m2))
                }
                _ => Err("mapa_definir(mapa, chave, valor)".to_string()),
            },
            "mapa_obter" => match args.as_slice() {
                [VmValor::Mapa(m), VmValor::Str(k)] => {
                    Ok(m.get(k).cloned().unwrap_or(VmValor::Null))
                }
                _ => Err("mapa_obter(mapa, chave)".to_string()),
            },
            "mapa_tem" => match args.as_slice() {
                [VmValor::Mapa(m), VmValor::Str(k)] => Ok(VmValor::Bool(m.contains_key(k))),
                _ => Err("mapa_tem(mapa, chave)".to_string()),
            },
            "mapa_remover" => match args.as_slice() {
                [VmValor::Mapa(m), VmValor::Str(k)] => {
                    let mut m2 = m.clone();
                    m2.remove(k);
                    Ok(VmValor::Mapa(m2))
                }
                _ => Err("mapa_remover(mapa, chave)".to_string()),
            },
            "mapa_chaves" => match vm_arg1(&args, "mapa_chaves")? {
                VmValor::Mapa(m) => {
                    let mut chaves: Vec<VmValor> = m.keys().cloned().map(VmValor::Str).collect();
                    chaves.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
                    Ok(VmValor::Lista(chaves))
                }
                v => Err(format!("'mapa_chaves' requer mapa, recebeu {}", v)),
            },
            "mapa_valores" => match vm_arg1(&args, "mapa_valores")? {
                VmValor::Mapa(m) => Ok(VmValor::Lista(m.values().cloned().collect())),
                v => Err(format!("'mapa_valores' requer mapa, recebeu {}", v)),
            },
            "remover" => match args.as_slice() {
                [VmValor::Lista(v), VmValor::Num(n)] => {
                    let mut v2 = v.clone();
                    let i = *n as usize;
                    if i < v2.len() {
                        v2.remove(i);
                        Ok(VmValor::Lista(v2))
                    } else {
                        Err(format!("Índice {} fora dos limites", i))
                    }
                }
                _ => Err("remover(lista, índice)".to_string()),
            },
            "html_escapar" => match vm_arg1(&args, "html_escapar")? {
                VmValor::Str(s) => Ok(VmValor::Str(
                    s.replace('&', "&amp;")
                        .replace('<', "&lt;")
                        .replace('>', "&gt;"),
                )),
                v => Ok(VmValor::Str(
                    v.to_string()
                        .replace('&', "&amp;")
                        .replace('<', "&lt;")
                        .replace('>', "&gt;"),
                )),
            },
            "substituir" => match args.as_slice() {
                [VmValor::Str(t), VmValor::Str(de), VmValor::Str(para)] => {
                    Ok(VmValor::Str(t.replace(de.as_str(), para.as_str())))
                }
                _ => Err("substituir(texto, de, para)".to_string()),
            },
            "dividir" => match args.as_slice() {
                [VmValor::Str(t), VmValor::Str(sep)] => Ok(VmValor::Lista(
                    t.split(sep.as_str())
                        .map(|s| VmValor::Str(s.to_string()))
                        .collect(),
                )),
                _ => Err("dividir(texto, sep)".to_string()),
            },
            "juntar" => match args.as_slice() {
                [VmValor::Lista(v), VmValor::Str(sep)] => Ok(VmValor::Str(
                    v.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(sep),
                )),
                _ => Err("juntar(lista, sep)".to_string()),
            },
            "contem_texto" => match args.as_slice() {
                [VmValor::Str(t), VmValor::Str(s)] => Ok(VmValor::Bool(t.contains(s.as_str()))),
                _ => Err("contem_texto(texto, sub)".to_string()),
            },
            "__store__" => {
                if args.len() == 2 {
                    if let VmValor::Str(nome) = &args[0] {
                        vars.insert(Arc::from(nome.as_str()), args[1].clone());
                    }
                }
                Ok(VmValor::Null)
            }
            // ── Tensores (criação e consulta) ─────────────────────────────
            "mat_de" => {
                // mat_de([[1,2],[3,4]])
                let v = vm_arg1(&args, "mat_de")?;
                match v {
                    VmValor::Lista(linhas) => {
                        let nlin = linhas.len();
                        if nlin == 0 {
                            return Ok(VmValor::Tensor {
                                shape: vec![0, 0],
                                dados: Arc::new(Vec::new()),
                            });
                        }
                        let ncol = match &linhas[0] {
                            VmValor::Lista(r) => r.len(),
                            _ => return Err("mat_de: cada linha deve ser lista".to_string()),
                        };
                        let mut dados = Vec::with_capacity(nlin * ncol);
                        for linha in linhas {
                            match linha {
                                VmValor::Lista(r) => {
                                    for v in r {
                                        match v {
                                            VmValor::Num(x) => dados.push(*x),
                                            _ => {
                                                return Err("mat_de: elementos devem ser numeros"
                                                    .to_string())
                                            }
                                        }
                                    }
                                }
                                _ => return Err("mat_de: cada linha deve ser lista".to_string()),
                            }
                        }
                        Ok(VmValor::Tensor {
                            shape: vec![nlin, ncol],
                            dados: Arc::new(dados),
                        })
                    }
                    _ => Err(format!("mat_de: requer lista de listas, recebeu {}", v)),
                }
            }
            "tensor_de" => {
                if args.len() < 2 {
                    return Err("tensor_de(dados, shape)".to_string());
                }
                let shape: Vec<usize> = match &args[1] {
                    VmValor::Lista(l) => l
                        .iter()
                        .map(|v| match v {
                            VmValor::Num(n) => Ok(*n as usize),
                            VmValor::Int(n) => Ok(*n as usize),
                            _ => Err("tensor_de: shape deve ser lista de inteiros".to_string()),
                        })
                        .collect::<Result<_, _>>()?,
                    _ => return Err("tensor_de: shape deve ser lista".to_string()),
                };
                let dados: Vec<f64> = match &args[0] {
                    VmValor::Lista(l) => l
                        .iter()
                        .map(|v| match v {
                            VmValor::Num(n) => Ok(*n),
                            VmValor::Int(n) => Ok(*n as f64),
                            _ => Err("tensor_de: dados devem ser numeros".to_string()),
                        })
                        .collect::<Result<_, _>>()?,
                    _ => return Err("tensor_de: dados deve ser lista de numeros".to_string()),
                };
                let esperado: usize = shape.iter().product();
                if dados.len() != esperado {
                    return Err(format!(
                        "tensor_de: {} elementos mas shape {:?} requer {}",
                        dados.len(),
                        shape,
                        esperado
                    ));
                }
                Ok(VmValor::Tensor {
                    shape,
                    dados: Arc::new(dados),
                })
            }
            "tensor_zeros" => {
                let shape = vm_shape_de_args(&args, "tensor_zeros")?;
                let n: usize = shape.iter().product();
                Ok(VmValor::Tensor {
                    shape,
                    dados: Arc::new(vec![0.0; n]),
                })
            }
            "tensor_uns" => {
                let shape = vm_shape_de_args(&args, "tensor_uns")?;
                let n: usize = shape.iter().product();
                Ok(VmValor::Tensor {
                    shape,
                    dados: Arc::new(vec![1.0; n]),
                })
            }
            "tensor_shape" => match vm_arg1(&args, "tensor_shape")? {
                VmValor::Tensor { shape, .. } => Ok(VmValor::Lista(
                    shape.iter().map(|&d| VmValor::Num(d as f64)).collect(),
                )),
                v => Err(format!("tensor_shape: requer tensor, recebeu {}", v)),
            },
            "tensor_ndim" => match vm_arg1(&args, "tensor_ndim")? {
                VmValor::Tensor { shape, .. } => Ok(VmValor::Num(shape.len() as f64)),
                v => Err(format!("tensor_ndim: requer tensor, recebeu {}", v)),
            },
            "tensor_tamanho" => match vm_arg1(&args, "tensor_tamanho")? {
                VmValor::Tensor { dados, .. } => Ok(VmValor::Num(dados.len() as f64)),
                v => Err(format!("tensor_tamanho: requer tensor, recebeu {}", v)),
            },
            "tensor_reshape" => {
                if args.len() < 2 {
                    return Err("tensor_reshape(tensor, shape)".to_string());
                }
                let nova_shape: Vec<usize> = match &args[1] {
                    VmValor::Lista(l) => l
                        .iter()
                        .map(|v| match v {
                            VmValor::Num(n) => Ok(*n as usize),
                            _ => Err("shape deve ser lista de inteiros".to_string()),
                        })
                        .collect::<Result<_, _>>()?,
                    _ => return Err("tensor_reshape: shape deve ser lista".to_string()),
                };
                match &args[0] {
                    VmValor::Tensor { dados, .. } => {
                        let esperado: usize = nova_shape.iter().product();
                        if dados.len() != esperado {
                            return Err(format!("tensor_reshape: tamanho incompativel"));
                        }
                        Ok(VmValor::Tensor {
                            shape: nova_shape,
                            dados: dados.clone(),
                        })
                    }
                    v => Err(format!("tensor_reshape: requer tensor, recebeu {}", v)),
                }
            }
            "mat_para_lista" | "tensor_para_lista" => match vm_arg1(&args, "tensor_para_lista")? {
                VmValor::Tensor { shape, dados } => Ok(vm_tensor_para_lista(shape, dados)),
                v => Err(format!("tensor_para_lista: requer tensor, recebeu {}", v)),
            },

            // ── Roteamento HTTP — handler deve permanecer como VmValor::Funcao ──
            "rota" => {
                if args.len() != 3 {
                    return Err("'rota' requer 3 argumentos: metodo, padrao, handler".to_string());
                }
                let metodo = match &args[0] {
                    VmValor::Str(s) => s.to_uppercase(),
                    v => return Err(format!("'rota' metodo deve ser texto, recebeu {}", v)),
                };
                let padrao = match args[1].clone() {
                    VmValor::Str(s) => s,
                    v => return Err(format!("'rota' caminho deve ser texto, recebeu {}", v)),
                };
                let handler = args.into_iter().nth(2).unwrap();
                match &handler {
                    VmValor::Funcao { .. } => {}
                    v => return Err(format!("'rota' handler deve ser funcao, recebeu {}", v)),
                }
                crate::servidor::registrar_rota(metodo, padrao, handler);
                Ok(VmValor::Null)
            }

            // ── Primitivos web ────────────────────────────────────────────────

            // Mapa global de aplicação (thread-safe, com TTL opcional)
            "global_definir" => {
                if args.len() < 2 {
                    return Err(
                        "global_definir(chave, valor) ou global_definir(chave, valor, ttl_seg)"
                            .to_string(),
                    );
                }
                let chave = match args.remove(0) {
                    VmValor::Str(s) => s,
                    v => {
                        return Err(format!(
                            "global_definir: chave deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let valor = args.remove(0);
                let ttl = if !args.is_empty() {
                    Some(
                        vm_para_f64(&args[0])
                            .map(|n| n as u64)
                            .map_err(|e| format!("global_definir: {}", e))?,
                    )
                } else {
                    None
                };
                crate::servidor::global_definir(chave, valor, ttl);
                Ok(VmValor::Null)
            }
            "global_obter" => {
                let chave = match vm_arg1(&args, "global_obter")? {
                    VmValor::Str(s) => s.clone(),
                    v => return Err(format!("global_obter: chave deve ser texto, recebeu {}", v)),
                };
                Ok(crate::servidor::global_obter(&chave).unwrap_or(VmValor::Null))
            }
            "global_apagar" => {
                let chave = match vm_arg1(&args, "global_apagar")? {
                    VmValor::Str(s) => s.clone(),
                    v => {
                        return Err(format!(
                            "global_apagar: chave deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                crate::servidor::global_apagar(&chave);
                Ok(VmValor::Null)
            }
            "global_existe" => {
                let chave = match vm_arg1(&args, "global_existe")? {
                    VmValor::Str(s) => s.clone(),
                    v => {
                        return Err(format!(
                            "global_existe: chave deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                Ok(VmValor::Bool(crate::servidor::global_existe(&chave)))
            }
            "global_listar" => {
                let lista = crate::servidor::global_listar();
                Ok(VmValor::Lista(
                    lista.into_iter().map(VmValor::Str).collect(),
                ))
            }

            // Resposta HTTP controlada pelo PEP
            "definir_status" => {
                let s = vm_para_f64(vm_arg1(&args, "definir_status")?)
                    .map(|n| n as u16)
                    .map_err(|e| format!("definir_status: {}", e))?;
                crate::interpretador::http_definir_status(s);
                Ok(VmValor::Null)
            }
            "definir_cabecalho" => {
                if args.len() < 2 {
                    return Err("definir_cabecalho(nome, valor)".to_string());
                }
                let k = match &args[0] {
                    VmValor::Str(s) => s.clone(),
                    v => {
                        return Err(format!(
                            "definir_cabecalho: nome deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let v = args[1].to_string();
                crate::interpretador::http_definir_cabecalho(k, v);
                Ok(VmValor::Null)
            }
            "resposta" => {
                // resposta(corpo, status)
                // resposta(corpo, status, headers_mapa)
                if args.len() < 2 {
                    return Err("resposta(corpo, status)".to_string());
                }
                let status = vm_para_f64(&args[1])
                    .map(|n| n as u16)
                    .map_err(|e| format!("resposta: status deve ser número — {}", e))?;
                if let Some(VmValor::Mapa(hdrs)) = args.get(2) {
                    for (k, v) in hdrs {
                        crate::interpretador::http_definir_cabecalho(k.clone(), v.to_string());
                    }
                }
                let corpo: Vec<u8> = match &args[0] {
                    VmValor::Bytes(b) => b.as_ref().clone(),
                    VmValor::Str(s) => s.as_bytes().to_vec(),
                    v => v.to_string().into_bytes(),
                };
                crate::interpretador::http_limpar_corpo();
                crate::interpretador::http_escrever_corpo(&corpo);
                crate::interpretador::http_definir_status(status);
                Ok(VmValor::Null)
            }

            // Log para stderr (não polui o corpo HTTP)
            "log" => {
                let partes: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                eprintln!("{}", partes.join(" "));
                Ok(VmValor::Null)
            }

            // Middleware
            "usar" => {
                let handler = vm_arg1(&args, "usar")?.clone();
                match &handler {
                    VmValor::Funcao { .. } => {}
                    v => return Err(format!("usar: requer funcao, recebeu {}", v)),
                }
                crate::servidor::registrar_middleware(handler);
                Ok(VmValor::Null)
            }
            "proximo" => {
                crate::servidor::sinalizar_proximo();
                Ok(VmValor::Null)
            }

            // SSE — Server-Sent Events
            "sse_iniciar" => {
                crate::servidor::sse_iniciar();
                Ok(VmValor::Null)
            }
            "sse_enviar" => {
                let v = vm_arg1(&args, "sse_enviar")?.clone();
                crate::servidor::sse_enviar(&v);
                Ok(VmValor::Null)
            }
            "sse_enviar_evento" => {
                if args.len() < 2 {
                    return Err("sse_enviar_evento(evento, dados)".to_string());
                }
                let evento = match &args[0] {
                    VmValor::Str(s) => s.clone(),
                    v => {
                        return Err(format!(
                            "sse_enviar_evento: evento deve ser texto, recebeu {}",
                            v
                        ))
                    }
                };
                let dados = args[1].clone();
                crate::servidor::sse_enviar_evento(&evento, &dados);
                Ok(VmValor::Null)
            }
            "sse_fechar" => {
                crate::servidor::sse_fechar();
                Ok(VmValor::Null)
            }

            // Tarefa em segundo plano (fire-and-forget)
            "executar_segundo_plano" => {
                let handler = vm_arg1(&args, "executar_segundo_plano")?.clone();
                match &handler {
                    VmValor::Funcao { .. } => {}
                    v => {
                        return Err(format!(
                            "executar_segundo_plano: requer funcao, recebeu {}",
                            v
                        ))
                    }
                }
                std::thread::spawn(move || {
                    let mut vm = crate::vm::Maquina::nova();
                    let _ = vm.chamar_funcao(handler, vec![]);
                });
                Ok(VmValor::Null)
            }

            // ── Autodiff reverso (grad_*) ─────────────────────────────────────
            "grad_ativar" => {
                crate::autodiff::ativar();
                Ok(VmValor::Null)
            }
            "grad_desativar" => {
                crate::autodiff::desativar();
                Ok(VmValor::Null)
            }
            "grad_limpar" => {
                crate::autodiff::limpar();
                Ok(VmValor::Null)
            }
            "grad_zerar_todos" => {
                crate::autodiff::zerar_todos();
                Ok(VmValor::Null)
            }

            "grad_zerar" => match vm_arg1(&args, "grad_zerar")? {
                VmValor::TensorGrad { id, .. } => {
                    crate::autodiff::zerar_grad(*id);
                    Ok(VmValor::Null)
                }
                v => Err(format!("grad_zerar: requer tensor_grad, recebeu {}", v)),
            },
            "grad_tensor" => {
                // Registra um tensor como folha do grafo de computação.
                // Se autodiff não estiver ativo, retorna o tensor sem modificação.
                match vm_arg1(&args, "grad_tensor")? {
                    VmValor::Tensor { shape, dados } => {
                        let id = crate::autodiff::novo_tensor(shape.clone(), dados.clone());
                        Ok(VmValor::TensorGrad {
                            shape: shape.clone(),
                            dados: dados.clone(),
                            id,
                        })
                    }
                    VmValor::TensorGrad { shape, dados, id } => Ok(VmValor::TensorGrad {
                        shape: shape.clone(),
                        dados: dados.clone(),
                        id: *id,
                    }),
                    v => Err(format!("grad_tensor: requer tensor, recebeu {}", v)),
                }
            }
            "grad_de" => {
                // Retorna o gradiente acumulado de um TensorGrad como Tensor normal.
                match vm_arg1(&args, "grad_de")? {
                    VmValor::TensorGrad { id, .. } => match crate::autodiff::obter_grad(*id) {
                        Some((shape, dados)) => Ok(VmValor::Tensor {
                            shape,
                            dados: Arc::new(dados),
                        }),
                        None => Err(format!("grad_de: ID {} não encontrado na fita", id)),
                    },
                    v => Err(format!("grad_de: requer tensor_grad, recebeu {}", v)),
                }
            }
            "grad_retropropagar" => match vm_arg1(&args, "grad_retropropagar")? {
                VmValor::TensorGrad { id, .. } => {
                    crate::autodiff::retropropagar(*id)?;
                    Ok(VmValor::Null)
                }
                v => Err(format!(
                    "grad_retropropagar: requer tensor_grad (resultado da perda), recebeu {}",
                    v
                )),
            },
            // Operações binárias com rastreamento
            "grad_soma" | "grad_sub" | "grad_mul" | "grad_matmul" => {
                if args.len() != 2 {
                    return Err(format!("{}: requer 2 argumentos", nome));
                }
                let (a, b) = vm_desempilhar_dois_grad(nome, args)?;
                let (sa, da, ia) = a;
                let (sb, db, ib) = b;
                let (s, d, id) = match nome {
                    "grad_soma" => crate::autodiff::op_soma(sa, da, ia, sb, db, ib)?,
                    "grad_sub" => crate::autodiff::op_sub(sa, da, ia, sb, db, ib)?,
                    "grad_mul" => crate::autodiff::op_mul(sa, da, ia, sb, db, ib)?,
                    "grad_matmul" => crate::autodiff::op_matmul(sa, da, ia, sb, db, ib)?,
                    _ => unreachable!(),
                };
                Ok(VmValor::TensorGrad {
                    shape: s,
                    dados: d,
                    id,
                })
            }
            // Operações unárias com rastreamento
            "grad_relu" | "grad_sigmoid" | "grad_tanh" | "grad_exp" | "grad_log" | "grad_neg" => {
                let (s, d, i) = vm_desempilhar_um_grad(nome, args)?;
                let (s2, d2, id) = match nome {
                    "grad_relu" => crate::autodiff::op_relu(s, d, i)?,
                    "grad_sigmoid" => crate::autodiff::op_sigmoid(s, d, i)?,
                    "grad_tanh" => crate::autodiff::op_tanh(s, d, i)?,
                    "grad_exp" => crate::autodiff::op_exp(s, d, i)?,
                    "grad_log" => crate::autodiff::op_log(s, d, i)?,
                    "grad_neg" => crate::autodiff::op_neg(s, d, i)?,
                    _ => unreachable!(),
                };
                Ok(VmValor::TensorGrad {
                    shape: s2,
                    dados: d2,
                    id,
                })
            }
            "grad_soma_total" => {
                let (s, d, i) = vm_desempilhar_um_grad("grad_soma_total", args)?;
                let (s2, d2, id) = crate::autodiff::op_soma_total(s, d, i)?;
                Ok(VmValor::TensorGrad {
                    shape: s2,
                    dados: d2,
                    id,
                })
            }
            "grad_escalar_mul" => {
                if args.len() != 2 {
                    return Err("grad_escalar_mul(tensor_grad, escalar)".to_string());
                }
                let (s, d, i) = vm_grad_extrair_um(args.remove(0), "grad_escalar_mul")?;
                let esc = vm_para_f64(&args[0]).map_err(|e| format!("grad_escalar_mul: {}", e))?;
                let (s2, d2, id) = crate::autodiff::op_escalar_mul(s, d, i, esc)?;
                Ok(VmValor::TensorGrad {
                    shape: s2,
                    dados: d2,
                    id,
                })
            }
            "grad_dados" => {
                // Retorna os dados (forward values) de um TensorGrad como Tensor normal.
                match vm_arg1(&args, "grad_dados")? {
                    VmValor::TensorGrad { shape, dados, .. } => Ok(VmValor::Tensor {
                        shape: shape.clone(),
                        dados: dados.clone(),
                    }),
                    VmValor::Tensor { shape, dados } => Ok(VmValor::Tensor {
                        shape: shape.clone(),
                        dados: dados.clone(),
                    }),
                    v => Err(format!(
                        "grad_dados: requer tensor ou tensor_grad, recebeu {}",
                        v
                    )),
                }
            }

            // ── API de dispositivo (CPU/GPU) ─────────────────────────────────
            "tensor_dispositivo" => match vm_arg1(&args, "tensor_dispositivo")? {
                VmValor::Tensor { dados, .. } | VmValor::TensorGrad { dados, .. } => {
                    Ok(VmValor::Str(obter_dispositivo(dados).to_string()))
                }
                v => Err(format!("tensor_dispositivo: requer tensor, recebeu {}", v)),
            },
            "tensor_para_gpu" => {
                // tensor_para_gpu(tensor) → mesmo tensor marcado como cuda:0
                // tensor_para_gpu(tensor, n) → cuda:n
                // Com --features gpu-cuda, move dados para a GPU de verdade.
                let (shape, dados) = match vm_arg1(&args, "tensor_para_gpu")? {
                    VmValor::Tensor { shape, dados } | VmValor::TensorGrad { shape, dados, .. } => {
                        (shape.clone(), dados.clone())
                    }
                    v => return Err(format!("tensor_para_gpu: requer tensor, recebeu {}", v)),
                };
                let gpu_idx: u32 = if args.len() >= 2 {
                    vm_para_f64(&args[1]).map(|n| n as u32).unwrap_or(0)
                } else {
                    0
                };
                registrar_dispositivo(&dados, Dispositivo::Cuda(gpu_idx));
                #[cfg(feature = "gpu-cuda")]
                {
                    // Com candle: mover dados para tensor CUDA real
                    // Placeholder para implementação futura com candle-core
                    // use candle_core::{Device, Tensor as CandleTensor};
                    // let dev = Device::new_cuda(gpu_idx as usize)?;
                    // ...
                    return Err(
                        "GPU CUDA: implementação candle não compilada nesta build".to_string()
                    );
                }
                Ok(VmValor::Tensor { shape, dados })
            }
            "tensor_para_metal" => {
                let (shape, dados) = match vm_arg1(&args, "tensor_para_metal")? {
                    VmValor::Tensor { shape, dados } | VmValor::TensorGrad { shape, dados, .. } => {
                        (shape.clone(), dados.clone())
                    }
                    v => return Err(format!("tensor_para_metal: requer tensor, recebeu {}", v)),
                };
                registrar_dispositivo(&dados, Dispositivo::Metal);
                #[cfg(feature = "gpu-metal")]
                {
                    return Err(
                        "GPU Metal: implementação candle não compilada nesta build".to_string()
                    );
                }
                Ok(VmValor::Tensor { shape, dados })
            }
            "tensor_para_cpu" => {
                // Move tensor de volta para CPU (no-op com CPU-only build)
                match vm_arg1(&args, "tensor_para_cpu")? {
                    VmValor::Tensor { shape, dados } | VmValor::TensorGrad { shape, dados, .. } => {
                        registrar_dispositivo(dados, Dispositivo::Cpu);
                        Ok(VmValor::Tensor {
                            shape: shape.clone(),
                            dados: dados.clone(),
                        })
                    }
                    v => Err(format!("tensor_para_cpu: requer tensor, recebeu {}", v)),
                }
            }

            nome => {
                // Bridge: converte args VmValor→Valor, chama nativa do interpretador, converte resultado de volta
                let interp_args: Vec<crate::interpretador::Valor> =
                    args.into_iter().map(vm_para_valor).collect();
                self.nativas
                    .chamar_nativa_vm(nome, interp_args, HashMap::new())
                    .map(valor_para_vm)
            }
        }
    }
}

// ── Auxiliares para funções grad_* ───────────────────────────────────────────

fn vm_grad_extrair_um(v: VmValor, ctx: &str) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    match v {
        VmValor::TensorGrad { shape, dados, id } => Ok((shape, dados, id)),
        VmValor::Tensor { shape, dados } => {
            // Folha implícita — cria ID mesmo que não esteja gravando
            let id = crate::autodiff::novo_tensor(shape.clone(), dados.clone());
            Ok((shape, dados, id))
        }
        o => Err(format!(
            "{}: requer tensor ou tensor_grad, recebeu {}",
            ctx, o
        )),
    }
}

fn vm_desempilhar_um_grad(
    ctx: &str,
    args: Vec<VmValor>,
) -> Result<(Vec<usize>, Arc<Vec<f64>>, usize), String> {
    if args.len() != 1 {
        return Err(format!("{}: requer 1 argumento", ctx));
    }
    let mut a = args;
    vm_grad_extrair_um(a.remove(0), ctx)
}

fn vm_desempilhar_dois_grad(
    ctx: &str,
    args: Vec<VmValor>,
) -> Result<
    (
        (Vec<usize>, Arc<Vec<f64>>, usize),
        (Vec<usize>, Arc<Vec<f64>>, usize),
    ),
    String,
> {
    if args.len() != 2 {
        return Err(format!("{}: requer 2 argumentos", ctx));
    }
    let mut a = args;
    let b = vm_grad_extrair_um(a.remove(1), ctx)?;
    let aa = vm_grad_extrair_um(a.remove(0), ctx)?;
    Ok((aa, b))
}

// ── Conversões VmValor ↔ Valor ────────────────────────────────────────────────

pub fn vm_para_valor(v: VmValor) -> crate::interpretador::Valor {
    use crate::interpretador::Valor;
    match v {
        VmValor::Int(n) => Valor::Inteiro(n),
        VmValor::Num(n) => Valor::Numero(n),
        VmValor::Str(s) => Valor::Texto(s),
        VmValor::Bool(b) => Valor::Booleano(b),
        VmValor::Null => Valor::Nulo,
        VmValor::Lista(l) => Valor::Lista(l.into_iter().map(vm_para_valor).collect()),
        VmValor::Mapa(m) => {
            Valor::Mapa(m.into_iter().map(|(k, v)| (k, vm_para_valor(v))).collect())
        }
        VmValor::Tensor { shape, dados } => Valor::Tensor { shape, dados },
        // TensorGrad converte para Tensor normal (perde o grad-ID — comportamento correto para o bridge)
        VmValor::TensorGrad { shape, dados, .. } => Valor::Tensor { shape, dados },
        VmValor::Bytes(b) => Valor::Bytes(b),
        VmValor::ConexaoBD(id) => Valor::ConexaoBD(id),
        VmValor::ConexaoSQLite(id) => Valor::ConexaoSQLite(id),
        VmValor::Erro {
            tipo,
            mensagem,
            pilha,
        } => Valor::Erro {
            tipo,
            mensagem,
            pilha,
        },
        // bytecode não pode ser convertido para AST — retorna nulo para não quebrar o bridge
        VmValor::Funcao { .. } => Valor::Nulo,
    }
}

pub fn valor_para_vm(v: crate::interpretador::Valor) -> VmValor {
    use crate::interpretador::Valor;
    match v {
        Valor::Inteiro(n) => VmValor::Int(n),
        Valor::Numero(n) => VmValor::Num(n),
        Valor::Texto(s) => VmValor::Str(s),
        Valor::Booleano(b) => VmValor::Bool(b),
        Valor::Nulo => VmValor::Null,
        Valor::Lista(l) => VmValor::Lista(l.into_iter().map(valor_para_vm).collect()),
        Valor::Mapa(m) => {
            VmValor::Mapa(m.into_iter().map(|(k, v)| (k, valor_para_vm(v))).collect())
        }
        Valor::Tensor { shape, dados } => VmValor::Tensor { shape, dados },
        Valor::Bytes(b) => VmValor::Bytes(b),
        Valor::ConexaoBD(id) => VmValor::ConexaoBD(id),
        Valor::ConexaoSQLite(id) => VmValor::ConexaoSQLite(id),
        Valor::Erro {
            tipo,
            mensagem,
            pilha,
        } => VmValor::Erro {
            tipo,
            mensagem,
            pilha,
        },
        Valor::Funcao { .. } | Valor::FuncaoNativa(_) => VmValor::Null,
    }
}

// ── Operações tensoriais da VM ────────────────────────────────────────────────

/// Extrai shape+dados de Tensor OU TensorGrad (descartando o grad-ID).
/// Ops de bytecode normais não rastreiam gradientes — se precisar de autodiff, use grad_*.
fn extrair_tensor(v: VmValor, ctx: &str) -> Result<(Vec<usize>, Arc<Vec<f64>>), String> {
    match v {
        VmValor::Tensor { shape, dados } => Ok((shape, dados)),
        VmValor::TensorGrad { shape, dados, .. } => Ok((shape, dados)),
        o => Err(format!("{}: requer tensor, recebeu {}", ctx, o)),
    }
}

fn vm_tensor_map(t: VmValor, f: impl Fn(f64) -> f64, ctx: &str) -> Result<VmValor, String> {
    let (shape, dados) = extrair_tensor(t, ctx)?;
    let nd: Vec<f64> = dados.iter().map(|&x| f(x)).collect();
    Ok(VmValor::Tensor {
        shape,
        dados: Arc::new(nd),
    })
}

fn vm_tensor_binop(
    a: VmValor,
    b: VmValor,
    f: impl Fn(f64, f64) -> f64,
    ctx: &str,
) -> Result<VmValor, String> {
    match (a, b) {
        (
            VmValor::Tensor {
                shape: sa,
                dados: da,
            }
            | VmValor::TensorGrad {
                shape: sa,
                dados: da,
                ..
            },
            VmValor::Tensor {
                shape: sb,
                dados: db,
            }
            | VmValor::TensorGrad {
                shape: sb,
                dados: db,
                ..
            },
        ) => {
            if sa != sb {
                return Err(format!(
                    "{}: shapes incompatíveis {:?} vs {:?}",
                    ctx, sa, sb
                ));
            }
            let nd: Vec<f64> = da.iter().zip(db.iter()).map(|(&x, &y)| f(x, y)).collect();
            Ok(VmValor::Tensor {
                shape: sa,
                dados: Arc::new(nd),
            })
        }
        (
            VmValor::Tensor { shape, dados } | VmValor::TensorGrad { shape, dados, .. },
            VmValor::Num(s),
        )
        | (
            VmValor::Num(s),
            VmValor::Tensor { shape, dados } | VmValor::TensorGrad { shape, dados, .. },
        ) => {
            let nd: Vec<f64> = dados.iter().map(|&x| f(x, s)).collect();
            Ok(VmValor::Tensor {
                shape,
                dados: Arc::new(nd),
            })
        }
        (a, b) => Err(format!("{}: tipos incompatíveis {} e {}", ctx, a, b)),
    }
}

fn vm_tensor_matmul(a: VmValor, b: VmValor) -> Result<VmValor, String> {
    let (sa, da) = extrair_tensor(a, "tensor_matmul")?;
    let (sb, db) = extrair_tensor(b, "tensor_matmul")?;
    if sa.len() != 2 || sb.len() != 2 {
        return Err("tensor_matmul: requer tensores 2D".to_string());
    }
    let (la, ca, lb, cb) = (sa[0], sa[1], sb[0], sb[1]);
    if ca != lb {
        return Err(format!(
            "tensor_matmul: shapes incompatíveis {}x{} · {}x{}",
            la, ca, lb, cb
        ));
    }
    use ndarray::ArrayView2;
    let av = ArrayView2::from_shape((la, ca), &da).map_err(|e| e.to_string())?;
    let bv = ArrayView2::from_shape((lb, cb), &db).map_err(|e| e.to_string())?;
    let c = av.dot(&bv);
    Ok(VmValor::Tensor {
        shape: vec![la, cb],
        dados: Arc::new(c.into_raw_vec()),
    })
}

fn vm_tensor_softmax(t: VmValor) -> Result<VmValor, String> {
    let (shape, dados) = extrair_tensor(t, "tensor_softmax")?;
    let max = dados.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = dados.iter().map(|&x| (x - max).exp()).collect();
    let soma: f64 = exps.iter().sum();
    let nd: Vec<f64> = exps.iter().map(|x| x / soma).collect();
    Ok(VmValor::Tensor {
        shape,
        dados: Arc::new(nd),
    })
}

fn vm_tensor_transpor(t: VmValor) -> Result<VmValor, String> {
    let (shape, dados) = extrair_tensor(t, "tensor_transpor")?;
    if shape.len() != 2 {
        return Err("tensor_transpor: requer tensor 2D".to_string());
    }
    let (nlin, ncol) = (shape[0], shape[1]);
    let mut nd = vec![0.0f64; nlin * ncol];
    for i in 0..nlin {
        for j in 0..ncol {
            nd[j * nlin + i] = dados[i * ncol + j];
        }
    }
    Ok(VmValor::Tensor {
        shape: vec![ncol, nlin],
        dados: Arc::new(nd),
    })
}

// ── Operações aritméticas gerais ──────────────────────────────────────────────

fn vm_add(a: VmValor, b: VmValor) -> Result<VmValor, String> {
    match (a, b) {
        (VmValor::Int(x), VmValor::Int(y)) => Ok(VmValor::Int(x.wrapping_add(y))),
        (VmValor::Num(x), VmValor::Num(y)) => Ok(VmValor::Num(x + y)),
        (VmValor::Int(x), VmValor::Num(y)) | (VmValor::Num(y), VmValor::Int(x)) => {
            Ok(VmValor::Num(x as f64 + y))
        }
        (VmValor::Str(x), y) => Ok(VmValor::Str(x + &y.to_string())),
        (x, VmValor::Str(y)) => Ok(VmValor::Str(x.to_string() + &y)),
        (a, b) => Err(format!("Não é possível somar {} e {}", a, b)),
    }
}
fn vm_mul(a: VmValor, b: VmValor) -> Result<VmValor, String> {
    match (a, b) {
        (VmValor::Int(x), VmValor::Int(y)) => Ok(VmValor::Int(x.wrapping_mul(y))),
        (VmValor::Num(x), VmValor::Num(y)) => Ok(VmValor::Num(x * y)),
        (VmValor::Int(x), VmValor::Num(y)) | (VmValor::Num(y), VmValor::Int(x)) => {
            Ok(VmValor::Num(x as f64 * y))
        }
        (VmValor::Str(s), VmValor::Num(n)) | (VmValor::Num(n), VmValor::Str(s)) => {
            Ok(VmValor::Str(s.repeat(n.max(0.0) as usize)))
        }
        (VmValor::Str(s), VmValor::Int(n)) | (VmValor::Int(n), VmValor::Str(s)) => {
            Ok(VmValor::Str(s.repeat(n.max(0) as usize)))
        }
        (a, b) => Err(format!("Não é possível multiplicar {} e {}", a, b)),
    }
}
fn vm_num(
    a: VmValor,
    b: VmValor,
    op: &str,
    f: impl Fn(f64, f64) -> f64,
) -> Result<VmValor, String> {
    match (a, b) {
        (VmValor::Int(x), VmValor::Int(y)) => Ok(VmValor::Num(f(x as f64, y as f64))),
        (VmValor::Num(x), VmValor::Num(y)) => Ok(VmValor::Num(f(x, y))),
        (VmValor::Int(x), VmValor::Num(y)) | (VmValor::Num(y), VmValor::Int(x)) => {
            Ok(VmValor::Num(f(x as f64, y)))
        }
        (a, b) => Err(format!("'{}' requer números, recebeu {} e {}", op, a, b)),
    }
}
fn vm_cmp(a: VmValor, b: VmValor, op: &str) -> Result<VmValor, String> {
    let cmp_num = |x: f64, y: f64| -> bool {
        match op {
            "<" => x < y,
            ">" => x > y,
            "<=" => x <= y,
            _ => x >= y,
        }
    };
    let cmp_int = |x: i64, y: i64| -> bool {
        match op {
            "<" => x < y,
            ">" => x > y,
            "<=" => x <= y,
            _ => x >= y,
        }
    };
    match (&a, &b) {
        (VmValor::Num(x), VmValor::Num(y)) => Ok(VmValor::Bool(cmp_num(*x, *y))),
        (VmValor::Int(x), VmValor::Int(y)) => Ok(VmValor::Bool(cmp_int(*x, *y))),
        (VmValor::Int(x), VmValor::Num(y)) => Ok(VmValor::Bool(cmp_num(*x as f64, *y))),
        (VmValor::Num(x), VmValor::Int(y)) => Ok(VmValor::Bool(cmp_num(*x, *y as f64))),
        (VmValor::Str(x), VmValor::Str(y)) => Ok(VmValor::Bool(match op {
            "<" => x < y,
            ">" => x > y,
            "<=" => x <= y,
            _ => x >= y,
        })),
        _ => Err(format!("Não é possível comparar {} e {}", a, b)),
    }
}
fn vm_get_index(obj: VmValor, idx: VmValor) -> Result<VmValor, String> {
    match (obj, idx) {
        (VmValor::Lista(v), VmValor::Num(n)) => {
            let i = n as usize;
            v.get(i)
                .cloned()
                .ok_or_else(|| format!("Índice {} fora dos limites", i))
        }
        (VmValor::Lista(v), VmValor::Int(n)) => {
            let i = n as usize;
            v.get(i)
                .cloned()
                .ok_or_else(|| format!("Índice {} fora dos limites", i))
        }
        (VmValor::Str(s), VmValor::Num(n)) => s
            .chars()
            .nth(n as usize)
            .map(|c| VmValor::Str(c.to_string()))
            .ok_or_else(|| format!("Índice {} fora dos limites", n as usize)),
        (VmValor::Str(s), VmValor::Int(n)) => s
            .chars()
            .nth(n as usize)
            .map(|c| VmValor::Str(c.to_string()))
            .ok_or_else(|| format!("Índice {} fora dos limites", n as usize)),
        (VmValor::Mapa(m), VmValor::Str(k)) => Ok(m.get(&k).cloned().unwrap_or(VmValor::Null)),
        (obj, idx) => Err(format!("Acesso inválido: {} com {}", obj, idx)),
    }
}
fn vm_set_index(obj: VmValor, idx: VmValor, val: VmValor) -> Result<VmValor, String> {
    match (obj, idx) {
        (VmValor::Lista(mut v), VmValor::Num(n)) => {
            let i = n as usize;
            if i < v.len() {
                v[i] = val;
                Ok(VmValor::Lista(v))
            } else {
                Err(format!("Índice {} fora dos limites", i))
            }
        }
        (VmValor::Lista(mut v), VmValor::Int(n)) => {
            let i = n as usize;
            if i < v.len() {
                v[i] = val;
                Ok(VmValor::Lista(v))
            } else {
                Err(format!("Índice {} fora dos limites", i))
            }
        }
        (VmValor::Mapa(mut m), VmValor::Str(k)) => {
            m.insert(k, val);
            Ok(VmValor::Mapa(m))
        }
        _ => Err("SetIndex inválido".to_string()),
    }
}
fn vm_arg1<'a>(args: &'a [VmValor], nome: &str) -> Result<&'a VmValor, String> {
    if args.len() != 1 {
        return Err(format!(
            "'{}' requer 1 argumento, recebeu {}",
            nome,
            args.len()
        ));
    }
    Ok(&args[0])
}
fn vm_shape_de_args(args: &[VmValor], nome: &str) -> Result<Vec<usize>, String> {
    if args.len() == 1 {
        match &args[0] {
            VmValor::Lista(l) => l
                .iter()
                .map(|v| match v {
                    VmValor::Num(n) => Ok(*n as usize),
                    _ => Err(format!("{}: shape deve ser lista de inteiros", nome)),
                })
                .collect(),
            VmValor::Num(n) => Ok(vec![*n as usize]),
            _ => Err(format!("{}: shape invalido", nome)),
        }
    } else {
        args.iter()
            .map(|v| match v {
                VmValor::Num(n) => Ok(*n as usize),
                _ => Err(format!("{}: args de shape devem ser inteiros", nome)),
            })
            .collect()
    }
}

fn vm_tensor_para_lista(shape: &[usize], dados: &[f64]) -> VmValor {
    if shape.len() == 1 {
        VmValor::Lista((0..shape[0]).map(|i| VmValor::Num(dados[i])).collect())
    } else {
        let stride: usize = shape[1..].iter().product();
        VmValor::Lista(
            (0..shape[0])
                .map(|i| vm_tensor_para_lista(&shape[1..], &dados[i * stride..(i + 1) * stride]))
                .collect(),
        )
    }
}

fn vm_para_f64(v: &VmValor) -> Result<f64, String> {
    match v {
        VmValor::Num(n) => Ok(*n),
        VmValor::Int(n) => Ok(*n as f64),
        v => Err(format!("esperado número, recebeu {}", v)),
    }
}

fn vm_para_usize(v: &VmValor, ctx: &str) -> Result<usize, String> {
    match v {
        VmValor::Int(n) if *n >= 0 => Ok(*n as usize),
        VmValor::Num(n) if *n >= 0.0 => Ok(*n as usize),
        v => Err(format!(
            "{}: eixo deve ser inteiro não-negativo, recebeu {}",
            ctx, v
        )),
    }
}

fn vm_tensor_reducao_global(
    t: VmValor,
    f: impl Fn(f64, f64) -> f64,
    init: f64,
    ctx: &str,
) -> Result<VmValor, String> {
    match t {
        VmValor::Tensor { dados, .. } => Ok(VmValor::Num(
            dados.iter().cloned().fold(init, |acc, x| f(acc, x)),
        )),
        v => Err(format!("{}: requer tensor, recebeu {}", ctx, v)),
    }
}

/// Redução ao longo de um eixo. Se `dividir_n` é true, divide pela dimensão (média).
fn vm_tensor_reducao_eixo(
    t: VmValor,
    eixo: usize,
    f: impl Fn(f64, f64) -> f64,
    init: f64,
    dividir_n: bool,
    ctx: &str,
) -> Result<VmValor, String> {
    match t {
        VmValor::Tensor { shape, dados } => {
            if eixo >= shape.len() {
                return Err(format!(
                    "{}: eixo {} fora dos limites para tensor {}D",
                    ctx,
                    eixo,
                    shape.len()
                ));
            }
            let n_eixo = shape[eixo];
            // stride deste eixo = produto das dimensões à direita
            let stride: usize = shape[eixo + 1..].iter().product::<usize>().max(1);
            // shape resultante = shape sem o eixo reduzido
            let mut nova_shape = shape.clone();
            nova_shape.remove(eixo);
            let n_out: usize = nova_shape.iter().product::<usize>().max(1);
            let mut out = vec![init; n_out];

            for (idx, &v) in dados.iter().enumerate() {
                // índice no output: remover contribuição do eixo
                let eixo_idx = (idx / stride) % n_eixo;
                let out_idx = idx / (stride * n_eixo) * stride + idx % stride;
                let _ = eixo_idx; // apenas para clareza; out_idx já é correto
                out[out_idx] = f(out[out_idx], v);
            }
            if dividir_n {
                let n = n_eixo as f64;
                for x in &mut out {
                    *x /= n;
                }
            }
            if nova_shape.is_empty() {
                Ok(VmValor::Num(out[0]))
            } else {
                Ok(VmValor::Tensor {
                    shape: nova_shape,
                    dados: Arc::new(out),
                })
            }
        }
        v => Err(format!("{}: requer tensor, recebeu {}", ctx, v)),
    }
}

fn vm_tensor_concatenar(lista: VmValor, eixo: usize) -> Result<VmValor, String> {
    let tensores: Vec<(Vec<usize>, Arc<Vec<f64>>)> = match lista {
        VmValor::Lista(l) => l
            .into_iter()
            .map(|v| match v {
                VmValor::Tensor { shape, dados } => Ok((shape, dados)),
                o => Err(format!("tensor_concatenar: elemento não é tensor: {}", o)),
            })
            .collect::<Result<_, _>>()?,
        v => {
            return Err(format!(
                "tensor_concatenar: requer lista de tensores, recebeu {}",
                v
            ))
        }
    };
    if tensores.is_empty() {
        return Err("tensor_concatenar: lista vazia".to_string());
    }
    let ref_shape = &tensores[0].0;
    if eixo >= ref_shape.len() {
        return Err(format!(
            "tensor_concatenar: eixo {} fora dos limites para tensor {}D",
            eixo,
            ref_shape.len()
        ));
    }
    for (s, _) in &tensores[1..] {
        if s.len() != ref_shape.len() {
            return Err("tensor_concatenar: tensores com ndim diferente".to_string());
        }
        for (i, (&a, &b)) in ref_shape.iter().zip(s.iter()).enumerate() {
            if i != eixo && a != b {
                return Err(format!(
                    "tensor_concatenar: shapes incompatíveis no eixo {}",
                    i
                ));
            }
        }
    }
    let mut nova_shape = ref_shape.clone();
    nova_shape[eixo] = tensores.iter().map(|(s, _)| s[eixo]).sum();
    let stride_out: usize = nova_shape[eixo + 1..].iter().product::<usize>().max(1);
    let n_out: usize = nova_shape.iter().product();
    let mut out = vec![0.0f64; n_out];
    let mut offset = 0usize;
    for (shape, dados) in &tensores {
        let n_bloco = shape[eixo];
        let outer: usize = shape[..eixo].iter().product::<usize>().max(1);
        for o in 0..outer {
            let src_base = o * n_bloco * stride_out;
            let dst_base = o * nova_shape[eixo] * stride_out + offset * stride_out;
            out[dst_base..dst_base + n_bloco * stride_out]
                .copy_from_slice(&dados[src_base..src_base + n_bloco * stride_out]);
        }
        offset += n_bloco;
    }
    Ok(VmValor::Tensor {
        shape: nova_shape,
        dados: Arc::new(out),
    })
}

fn vm_tensor_empilhar(lista: VmValor, eixo: usize) -> Result<VmValor, String> {
    let tensores: Vec<(Vec<usize>, Arc<Vec<f64>>)> = match lista {
        VmValor::Lista(l) => l
            .into_iter()
            .map(|v| match v {
                VmValor::Tensor { shape, dados } => Ok((shape, dados)),
                o => Err(format!("tensor_empilhar: elemento não é tensor: {}", o)),
            })
            .collect::<Result<_, _>>()?,
        v => {
            return Err(format!(
                "tensor_empilhar: requer lista de tensores, recebeu {}",
                v
            ))
        }
    };
    if tensores.is_empty() {
        return Err("tensor_empilhar: lista vazia".to_string());
    }
    let ref_shape = &tensores[0].0;
    for (s, _) in &tensores[1..] {
        if s != ref_shape {
            return Err("tensor_empilhar: todos os tensores devem ter o mesmo shape".to_string());
        }
    }
    let k = tensores.len();
    let n_elem: usize = ref_shape.iter().product::<usize>().max(1);
    let mut nova_shape = ref_shape.clone();
    if eixo > nova_shape.len() {
        return Err(format!("tensor_empilhar: eixo {} fora dos limites", eixo));
    }
    nova_shape.insert(eixo, k);
    let stride: usize = ref_shape[eixo..].iter().product::<usize>().max(1);
    let outer = n_elem / stride;
    let mut out = vec![0.0f64; k * n_elem];
    for (i, (_, dados)) in tensores.iter().enumerate() {
        for o in 0..outer {
            let src_base = o * stride;
            let dst_base = (o * k + i) * stride;
            out[dst_base..dst_base + stride].copy_from_slice(&dados[src_base..src_base + stride]);
        }
    }
    Ok(VmValor::Tensor {
        shape: nova_shape,
        dados: Arc::new(out),
    })
}

fn compilar_fonte_para_ops(abs: &Path, fonte: &str) -> Result<Vec<Op>, String> {
    let ext = abs.extension().and_then(|e| e.to_str()).unwrap_or("");
    let programa = if matches!(ext, "phtml" | "html" | "htm") {
        crate::template::compilar(fonte)?
    } else {
        let mut lex = crate::lexer::Lexer::novo(fonte);
        let tokens = lex.tokenizar()?;
        crate::parser::Parser::novo(tokens).parsear()?
    };
    crate::compilador::compilar(&programa)
}

fn vm_em(valor: &VmValor, colecao: &VmValor) -> Result<bool, String> {
    match colecao {
        VmValor::Lista(l) => Ok(l.contains(valor)),
        VmValor::Str(s) => match valor {
            VmValor::Str(sub) => Ok(s.contains(sub.as_str())),
            _ => Err("'em' com texto requer substrings".to_string()),
        },
        VmValor::Mapa(m) => match valor {
            VmValor::Str(k) => Ok(m.contains_key(k)),
            _ => Err("'em' com mapa requer chave texto".to_string()),
        },
        _ => Err(format!("'em' requer lista/texto/mapa, recebeu {}", colecao)),
    }
}
