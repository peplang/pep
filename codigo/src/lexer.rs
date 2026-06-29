/// Lexer da linguagem PEP  -  transforma codigo-fonte em tokens

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literais
    Inteiro(i64),
    Numero(f64),
    Texto(String),
    Verdadeiro,
    Falso,
    Nulo,

    // Identificadores e palavras-chave
    Identificador(String),
    Var,
    Se,
    Senao,
    Enquanto,
    Para,
    Funcao,
    Retornar,
    Imprimir,
    Pare,
    Continue,
    // Tratamento de erros
    Tentar,
    Capturar,
    Finalmente,
    Lancar,
    // Escolher/caso
    Escolher,
    Caso,
    Padrao,
    // Intervalo numerico e pertencimento
    De,
    Ate,
    Passo,
    Em,
    // Modulos
    Importar,
    Incluir,
    Requerer,
    Como,
    // Variadic
    PontoPontoPonto,

    // Operadores aritmeticos
    Mais,
    Menos,
    Estrela,
    Barra,
    BarraBarra,
    Percentual,

    // Operadores compostos (+=, -=, *=, /=, %=)
    MaisIgual,
    MenosIgual,
    EstrelaIgual,
    BarraIgual,
    PercentualIgual,

    // Operadores de comparacao
    Igual,
    DiferenteDe,
    MenorQue,
    MaiorQue,
    MenorOuIgual,
    MaiorOuIgual,

    // Operadores logicos
    E,
    Ou,
    Nao,

    // Atribuicao
    Atribuicao,

    // Delimitadores
    ParenEsq,
    ParenDir,
    ChaveEsq,
    ChaveDir,
    ColcheteEsq,
    ColcheteDir,
    Virgula,
    PontoEVirgula,
    Ponto,
    DoisPontos,

    // Operadores modernos
    /// `??`  — coalescência nula: `a ?? b`  (retorna b se a for nulo)
    NullCoalescente,
    /// `?.`  — acesso opcional: `a?.campo`  (retorna nulo se a for nulo)
    PontoOpcional,
    /// `=>`  — seta de funcao anonima: `x => x * 2`
    Seta,

    // Controle
    NovaLinha,
    FimDeArquivo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diferencia_inteiro_decimal_e_divisao_inteira() {
        let mut lexer = Lexer::novo("7 // 2\n1.5");
        let tokens = lexer.tokenizar().unwrap();
        assert_eq!(tokens[0].token, Token::Inteiro(7));
        assert_eq!(tokens[1].token, Token::BarraBarra);
        assert_eq!(tokens[2].token, Token::Inteiro(2));
        assert_eq!(tokens[4].token, Token::Numero(1.5));
    }
}

#[derive(Debug, Clone)]
pub struct TokenComPosicao {
    pub token: Token,
    pub linha: usize,
    #[allow(dead_code)]
    pub coluna: usize,
    pub contexto: String,
}

pub struct Lexer {
    fonte: Vec<char>,
    pos: usize,
    pub linha: usize,
    coluna: usize,
    linhas_fonte: Vec<String>,
}

impl Lexer {
    pub fn novo(codigo: &str) -> Self {
        Lexer {
            fonte: codigo.chars().collect(),
            pos: 0,
            linha: 1,
            coluna: 1,
            linhas_fonte: codigo.lines().map(str::to_string).collect(),
        }
    }

    fn contexto(&self, linha: usize) -> String {
        self.linhas_fonte
            .get(linha.saturating_sub(1))
            .cloned()
            .unwrap_or_default()
    }

    fn atual(&self) -> Option<char> {
        self.fonte.get(self.pos).copied()
    }
    fn avancar(&mut self) -> Option<char> {
        let c = self.fonte.get(self.pos).copied();
        self.pos += 1;
        if c == Some('\n') {
            self.linha += 1;
            self.coluna = 1;
        } else {
            self.coluna += 1;
        }
        c
    }

    fn pular_espacos(&mut self) {
        while matches!(
            self.atual(),
            Some(' ') | Some('\t') | Some('\r') | Some('\u{feff}')
        ) {
            self.avancar();
        }
    }

    fn ler_numero(&mut self) -> Token {
        // Hex literal: 0x... ou 0X...
        if self.pos + 1 < self.fonte.len()
            && self.fonte[self.pos] == '0'
            && matches!(self.fonte.get(self.pos + 1), Some('x') | Some('X'))
        {
            self.avancar(); // '0'
            self.avancar(); // 'x'
            let mut hex = String::new();
            while let Some(c) = self.atual() {
                if c.is_ascii_hexdigit() {
                    hex.push(c);
                    self.avancar();
                } else {
                    break;
                }
            }
            return Token::Inteiro(i64::from_str_radix(&hex, 16).unwrap_or(0));
        }
        // Decimal / Float
        let mut s = String::new();
        let mut tem_ponto = false;
        while let Some(c) = self.atual() {
            if c.is_ascii_digit() {
                s.push(c);
                self.avancar();
            } else if c == '.' && !tem_ponto {
                tem_ponto = true;
                s.push(c);
                self.avancar();
            } else {
                break;
            }
        }
        if tem_ponto {
            Token::Numero(s.parse().unwrap_or(0.0))
        } else {
            Token::Inteiro(s.parse().unwrap_or(0))
        }
    }

    fn ler_texto_ou_tripla(&mut self) -> Result<Token, String> {
        let delimitador = self.atual().unwrap_or('"');
        // Verifica se é string tripla (""" ou ''')
        if delimitador == '"' || delimitador == '\'' {
            let p1 = self.pos;
            // Olha os próximos 3 chars sem avançar
            if self.fonte.get(p1).copied() == Some(delimitador)
                && self.fonte.get(p1 + 1).copied() == Some(delimitador)
                && self.fonte.get(p1 + 2).copied() == Some(delimitador)
            {
                self.avancar();
                self.avancar();
                self.avancar(); // consome as 3 aspas
                return self.ler_texto_tripla(delimitador);
            }
        }
        self.ler_texto()
    }

    fn ler_texto_tripla(&mut self, delimitador: char) -> Result<Token, String> {
        let mut s = String::new();
        loop {
            // Verifica se encontrou fechamento """
            if self.atual() == Some(delimitador)
                && self.fonte.get(self.pos + 1).copied() == Some(delimitador)
                && self.fonte.get(self.pos + 2).copied() == Some(delimitador)
            {
                self.avancar();
                self.avancar();
                self.avancar();
                break;
            }
            match self.atual() {
                None => return Err(format!("String tripla nao fechada na linha {}", self.linha)),
                Some(c) => {
                    s.push(c);
                    self.avancar();
                }
            }
        }
        // Remove indentação comum (leading newline + trailing whitespace)
        let s = dedent_tripla(&s);
        Ok(Token::Texto(s))
    }

    fn ler_texto(&mut self) -> Result<Token, String> {
        let delimitador = self.avancar().unwrap_or('"');
        let mut s = String::new();
        loop {
            match self.atual() {
                None => return Err(format!("Texto nao fechado na linha {}", self.linha)),
                Some(c) if c == delimitador => {
                    self.avancar();
                    break;
                }
                Some('\\') => {
                    self.avancar();
                    match self.avancar() {
                        Some('n') => s.push('\n'),
                        Some('t') => s.push('\t'),
                        Some('"') => s.push('"'),
                        Some('\\') => s.push('\\'),
                        Some('\'') => s.push('\''),
                        Some('{') => s.push('{'),
                        Some('}') => s.push('}'),
                        Some(c) => {
                            s.push('\\');
                            s.push(c);
                        }
                        None => return Err("Sequencia de escape incompleta".to_string()),
                    }
                }
                Some(c) => {
                    s.push(c);
                    self.avancar();
                }
            }
        }
        Ok(Token::Texto(s))
    }

    fn ler_identificador(&mut self) -> Token {
        let mut s = String::new();
        while let Some(c) = self.atual() {
            if c.is_alphabetic() || c == '_' || c.is_ascii_digit() {
                s.push(c);
                self.avancar();
            } else {
                break;
            }
        }
        match s.as_str() {
            "var" => Token::Var,
            "se" => Token::Se,
            "senao" => Token::Senao,
            "enquanto" => Token::Enquanto,
            "para" => Token::Para,
            "em" => Token::Em,
            "funcao" => Token::Funcao,
            "retornar" => Token::Retornar,
            "imprimir" => Token::Imprimir,
            "verdadeiro" => Token::Verdadeiro,
            "falso" => Token::Falso,
            "nulo" => Token::Nulo,
            "e" => Token::E,
            "ou" => Token::Ou,
            "nao" => Token::Nao,
            "pare" => Token::Pare,
            "continue" => Token::Continue,
            "tentar" => Token::Tentar,
            "capturar" => Token::Capturar,
            "finalmente" => Token::Finalmente,
            "lancar" => Token::Lancar,
            "escolher" => Token::Escolher,
            "caso" => Token::Caso,
            "padrao" => Token::Padrao,
            "de" => Token::De,
            "ate" => Token::Ate,
            "passo" => Token::Passo,
            "importar" => Token::Importar,
            "incluir" => Token::Incluir,
            "requerer" => Token::Requerer,
            "como" => Token::Como,
            _ => Token::Identificador(s),
        }
    }

    pub fn tokenizar(&mut self) -> Result<Vec<TokenComPosicao>, String> {
        let mut tokens = Vec::new();
        loop {
            self.pular_espacos();
            let linha = self.linha;
            let coluna = self.coluna;
            let c = match self.atual() {
                None => {
                    tokens.push(TokenComPosicao {
                        token: Token::FimDeArquivo,
                        linha,
                        coluna,
                        contexto: self.contexto(linha),
                    });
                    break;
                }
                Some(c) => c,
            };

            if c == '#' {
                while self.atual().map_or(false, |c| c != '\n') {
                    self.avancar();
                }
                continue;
            }

            let token = match c {
                '\n' => {
                    self.avancar();
                    Token::NovaLinha
                }
                '0'..='9' => self.ler_numero(),
                '"' | '\'' => self.ler_texto_ou_tripla()?,
                c if c.is_alphabetic() || c == '_' => self.ler_identificador(),
                '+' => {
                    self.avancar();
                    if self.atual() == Some('=') {
                        self.avancar();
                        Token::MaisIgual
                    } else {
                        Token::Mais
                    }
                }
                '-' => {
                    self.avancar();
                    if self.atual() == Some('=') {
                        self.avancar();
                        Token::MenosIgual
                    } else {
                        Token::Menos
                    }
                }
                '*' => {
                    self.avancar();
                    if self.atual() == Some('=') {
                        self.avancar();
                        Token::EstrelaIgual
                    } else {
                        Token::Estrela
                    }
                }
                '/' => {
                    self.avancar();
                    if self.atual() == Some('/') {
                        self.avancar();
                        Token::BarraBarra
                    } else if self.atual() == Some('=') {
                        self.avancar();
                        Token::BarraIgual
                    } else {
                        Token::Barra
                    }
                }
                '%' => {
                    self.avancar();
                    if self.atual() == Some('=') {
                        self.avancar();
                        Token::PercentualIgual
                    } else {
                        Token::Percentual
                    }
                }
                '(' => {
                    self.avancar();
                    Token::ParenEsq
                }
                ')' => {
                    self.avancar();
                    Token::ParenDir
                }
                '{' => {
                    self.avancar();
                    Token::ChaveEsq
                }
                '}' => {
                    self.avancar();
                    Token::ChaveDir
                }
                '[' => {
                    self.avancar();
                    Token::ColcheteEsq
                }
                ']' => {
                    self.avancar();
                    Token::ColcheteDir
                }
                ',' => {
                    self.avancar();
                    Token::Virgula
                }
                ';' => {
                    self.avancar();
                    Token::PontoEVirgula
                }
                '.' => {
                    self.avancar();
                    if self.atual() == Some('.') {
                        self.avancar();
                        if self.atual() == Some('.') {
                            self.avancar();
                            Token::PontoPontoPonto
                        } else {
                            return Err(format!("Caractere inesperado '..' na linha {linha}"));
                        }
                    } else {
                        Token::Ponto
                    }
                }
                ':' => {
                    self.avancar();
                    Token::DoisPontos
                }
                '?' => {
                    self.avancar();
                    if self.atual() == Some('?') {
                        self.avancar();
                        Token::NullCoalescente
                    } else if self.atual() == Some('.') {
                        self.avancar();
                        Token::PontoOpcional
                    } else {
                        return Err(format!("Caractere inesperado '?' na linha {linha}"));
                    }
                }
                '=' => {
                    self.avancar();
                    if self.atual() == Some('=') {
                        self.avancar();
                        Token::Igual
                    } else if self.atual() == Some('>') {
                        self.avancar();
                        Token::Seta
                    } else {
                        Token::Atribuicao
                    }
                }
                '!' => {
                    self.avancar();
                    if self.atual() == Some('=') {
                        self.avancar();
                        Token::DiferenteDe
                    } else {
                        return Err(format!("Caractere inesperado '!' na linha {linha}"));
                    }
                }
                '<' => {
                    self.avancar();
                    if self.atual() == Some('=') {
                        self.avancar();
                        Token::MenorOuIgual
                    } else {
                        Token::MenorQue
                    }
                }
                '>' => {
                    self.avancar();
                    if self.atual() == Some('=') {
                        self.avancar();
                        Token::MaiorOuIgual
                    } else {
                        Token::MaiorQue
                    }
                }
                c => {
                    return Err(format!(
                        "Caractere desconhecido '{c}' na linha {linha}, coluna {coluna}"
                    ))
                }
            };
            tokens.push(TokenComPosicao {
                token,
                linha,
                coluna,
                contexto: self.contexto(linha),
            });
        }
        Ok(tokens)
    }
}

/// Remove indentação comum e newline inicial de strings triplas.
/// `"""\n    hello\n    world\n"""` → `"hello\nworld"`
fn dedent_tripla(s: &str) -> String {
    // Remove newline inicial se existir
    let s = s.strip_prefix('\n').unwrap_or(s);
    let s = s.strip_prefix("\r\n").unwrap_or(s);

    let linhas: Vec<&str> = s.lines().collect();
    if linhas.is_empty() {
        return s.to_string();
    }

    // Calcula indentação mínima (ignora linhas vazias)
    let min_indent = linhas
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);

    // Remove indentação comum e junta
    let mut resultado = linhas
        .iter()
        .map(|l| {
            if l.len() >= min_indent {
                &l[min_indent..]
            } else {
                l.trim_start()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Remove newline/whitespace final
    while resultado.ends_with('\n') || resultado.ends_with(' ') {
        resultado.pop();
    }
    resultado
}
