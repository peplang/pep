/// Conjunto de instruções da VM de bytecode PEP
///
/// A VM é baseada em pilha. Cada frame de função tem:
///   - uma pilha de valores
///   - variáveis locais (HashMap)
///   - ponteiro de instrução (ip)
///
/// Endereços em Jump são índices absolutos no Vec<Op> do chunk atual.

#[derive(Debug, Clone)]
pub enum Op {
    // ── Literais ──────────────────────────────────────────────────────────────
    PushNum(f64),
    PushInt(i64),
    PushStr(String),
    PushBool(bool),
    PushNull,

    // ── Variáveis ─────────────────────────────────────────────────────────────
    // Arc<str> em vez de String: clone O(1) no loop de despacho da VM
    Load(std::sync::Arc<str>),
    Store(std::sync::Arc<str>),

    // ── Coleções ──────────────────────────────────────────────────────────────
    MakeList(usize),
    MakeMap(Vec<std::sync::Arc<str>>),
    GetIndex,
    SetIndex,

    // ── Aritmética ────────────────────────────────────────────────────────────
    Add,
    Sub,
    Mul,
    Div,
    IntDiv,
    Mod,
    Neg, // unário negativo
    Not, // unário não

    // ── Comparação e lógica ───────────────────────────────────────────────────
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,

    // ── Controle de fluxo (índices absolutos no Vec<Op>) ─────────────────────
    Jump(usize),      // pula incondicionalmente
    JumpFalse(usize), // pula se TOS == falso; desempilha
    JumpTrue(usize),  // pula se TOS == verdadeiro; desempilha

    // ── Funções ───────────────────────────────────────────────────────────────
    DefFunc {
        nome: std::sync::Arc<str>,
        params: Vec<std::sync::Arc<str>>,
        corpo: Vec<Op>,
    },
    Call(usize),
    CallNative(std::sync::Arc<str>, usize),
    Return, // desempilha valor de retorno e retorna do frame
    ReturnNull,

    // Modulos sempre sao compilados e executados pela propria VM.
    Include {
        caminho: String,
        obrigatorio: bool,
    },
    Import {
        caminho: String,
        alias: Option<String>,
    },

    // ── E/S ───────────────────────────────────────────────────────────────────
    Print(usize), // desempilha N valores, imprime com espaços e \n
    Write(usize), // desempilha N valores, escreve sem \n

    // ── Pertinência ───────────────────────────────────────────────────────────
    Em,    // TOS=coleção, TOS-1=valor → Bool (valor em coleção)
    NaoEm, // inverte Em

    // ── Loops numéricos otimizados (sem HashMap para o contador) ─────────────
    /// Pops passo, fim, ini da pilha; push Bool(ini ≤ fim em relação a passo);
    /// armazena contador interno (ini, fim, passo); Store var=ini se in_range.
    IterStart {
        var: std::sync::Arc<str>,
    },
    /// Incrementa o contador do topo; se ainda in_range: Store var=cur, Jump(loop_ini);
    /// senão: pop contador e cai no próximo op (fim do loop).
    IterNext {
        var: std::sync::Arc<str>,
        loop_ini: usize,
    },

    // ── Tratamento de erros ───────────────────────────────────────────────────
    Throw,           // lanca TOS; salta para catch handler ativo ou propaga
    TryCatch(usize), // registra handler catch em offset (backpatch)
    EndTry,          // remove handler catch mais recente

    // ── Opcodes tensoriais dedicados (evitam CallNative overhead) ────────────
    // Produto matricial
    TensorMatMul,   // pop B, pop A → push A·B  (ndarray SIMD)
    TensorTranspor, // pop T → push T.T (apenas 2D)
    // Ativações
    TensorReLU,    // pop T → push relu(T)
    TensorSigmoid, // pop T → push sigmoid(T)
    TensorSoftmax, // pop T → push softmax(T)
    TensorTanh,    // pop T → push tanh(T)
    // Aritmética element-wise (broadcast escalar ou tensor)
    TensorAdd,     // pop B, pop A → push A+B
    TensorSub,     // pop B, pop A → push A-B
    TensorMulElem, // pop B, pop A → push A*B  (element-wise, não matmul)
    TensorDivElem, // pop B, pop A → push A/B
    TensorPow,     // pop exp(escalar), pop A → push A^exp
    // Unárias
    TensorNeg,  // pop T → push -T
    TensorExp,  // pop T → push e^T
    TensorLog,  // pop T → push ln(T)
    TensorSqrt, // pop T → push sqrt(T)
    // Reduções globais (→ escalar)
    TensorSomaTotal,  // pop T → push soma de todos os elementos
    TensorMediaTotal, // pop T → push média
    TensorMaxTotal,   // pop T → push máximo
    TensorMinTotal,   // pop T → push mínimo
    // Reduções por eixo (→ tensor com dimensão removida)
    TensorSomaEixo,  // pop eixo(Int), pop T → push T.sum(axis=eixo)
    TensorMediaEixo, // pop eixo(Int), pop T → push T.mean(axis=eixo)
    TensorMaxEixo,   // pop eixo(Int), pop T → push T.max(axis=eixo)
    TensorMinEixo,   // pop eixo(Int), pop T → push T.min(axis=eixo)
    // Outras operações de forma
    TensorConcatenar, // pop eixo(Int), pop lista_de_tensores → push concatenado
    TensorEmpilhar,   // pop eixo(Int), pop lista_de_tensores → push empilhado (novo eixo)

    // ── Especial ──────────────────────────────────────────────────────────────
    Pop,  // descarta TOS
    Dup,  // duplica TOS
    Halt, // encerra execução
}
