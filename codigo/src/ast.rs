/// AST (Arvore Sintatica Abstrata) da linguagem PEP

// -- Parametro de funcao -------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Parametro {
    pub nome: String,
    pub padrao: Option<Expressao>,
    pub variadic: bool,
}

// -- Expressoes ----------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Expressao {
    Inteiro(i64),
    Numero(f64),
    Texto(String),
    Booleano(bool),
    Nulo,
    Lista(Vec<Expressao>),
    Mapa(Vec<(String, Expressao)>),

    Variavel(String),

    BinOp {
        esq: Box<Expressao>,
        op: OpBinario,
        dir: Box<Expressao>,
    },

    UnOp {
        op: OpUnario,
        expr: Box<Expressao>,
    },

    Atribuicao {
        nome: String,
        valor: Box<Expressao>,
    },

    AtribuicaoIndexada {
        objeto: Box<Expressao>,
        indice: Box<Expressao>,
        valor: Box<Expressao>,
    },

    ChamadaFuncao {
        nome: String,
        args: Vec<Expressao>,
    },

    Chamada {
        funcao: Box<Expressao>,
        args: Vec<Expressao>,
    },

    FuncaoAnonima {
        parametros: Vec<Parametro>,
        corpo: Vec<Instrucao>,
    },

    Acesso {
        objeto: Box<Expressao>,
        indice: Box<Expressao>,
    },

    /// `a?.campo` — retorna nulo se `a` for nulo, senao avalia `a.campo`
    AcessoOpcional {
        objeto: Box<Expressao>,
        chave: String,
    },

    /// `a ?? b` — retorna `b` se `a` for nulo
    NullCoalescente {
        esq: Box<Expressao>,
        dir: Box<Expressao>,
    },

    /// `x => expr` ou `(a, b) => expr` — função seta
    FuncaoSeta {
        parametros: Vec<String>,
        corpo: Box<Expressao>,
    },
}

// -- Operadores ----------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum OpBinario {
    Soma,
    Subtracao,
    Multiplicacao,
    Divisao,
    DivisaoInteira,
    Modulo,
    Igual,
    DiferenteDe,
    MenorQue,
    MaiorQue,
    MenorOuIgual,
    MaiorOuIgual,
    E,
    Ou,
    Em,
    NaoEm,
}

#[derive(Debug, Clone)]
pub enum OpUnario {
    Negativo,
    Nao,
}

// -- Instrucoes ----------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Instrucao {
    Localizada {
        linha: usize,
        contexto: String,
        instrucao: Box<Instrucao>,
    },

    Expressao(Expressao),

    DeclararVar {
        nome: String,
        valor: Option<Expressao>,
    },

    Se {
        condicao: Expressao,
        entao: Vec<Instrucao>,
        senao: Option<Vec<Instrucao>>,
    },

    Enquanto {
        condicao: Expressao,
        corpo: Vec<Instrucao>,
    },

    Para {
        variavel: String,
        iteravel: Expressao,
        corpo: Vec<Instrucao>,
    },

    // para i de 0 ate 10 [passo 2] { }
    ParaIntervalo {
        variavel: String,
        inicio: Expressao,
        fim: Expressao,
        passo: Option<Expressao>,
        corpo: Vec<Instrucao>,
    },

    // escolher expr { caso v1, v2 { } padrao { } }
    Escolher {
        expr: Expressao,
        casos: Vec<(Vec<Expressao>, Vec<Instrucao>)>,
        padrao: Option<Vec<Instrucao>>,
    },

    Funcao {
        nome: String,
        parametros: Vec<Parametro>,
        corpo: Vec<Instrucao>,
    },

    Retornar(Option<Expressao>),
    Pare,
    Continue,

    Imprimir(Vec<Expressao>),

    Tentar {
        corpo: Vec<Instrucao>,
        capturar: Option<(String, Vec<Instrucao>)>,
        finalmente: Option<Vec<Instrucao>>,
    },

    Lancar(Expressao),

    Importar {
        caminho: String,
        alias: Option<String>,
    },

    Incluir {
        caminho: String,
        obrigatorio: bool,
    },
}

pub type Programa = Vec<Instrucao>;
