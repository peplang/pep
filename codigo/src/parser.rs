/// Parser da linguagem PEP  -  transforma tokens em AST
use crate::ast::*;
use crate::lexer::{Lexer, Token, TokenComPosicao};

pub struct Parser {
    tokens: Vec<TokenComPosicao>,
    pos: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parseia_funcao_anonima() {
        let mut lexer = Lexer::novo("var dobro = funcao(x) { retornar x * 2 }");
        let tokens = lexer.tokenizar().unwrap();
        let mut parser = Parser::novo(tokens);
        let programa = parser.parsear().unwrap();
        let interna = match &programa[0] {
            Instrucao::Localizada { instrucao, .. } => instrucao.as_ref(),
            outra => outra,
        };
        match interna {
            Instrucao::DeclararVar {
                valor: Some(Expressao::FuncaoAnonima { parametros, .. }),
                ..
            } => {
                assert_eq!(parametros.len(), 1);
                assert_eq!(parametros[0].nome, "x");
                assert!(!parametros[0].variadic);
                assert!(parametros[0].padrao.is_none());
            }
            _ => panic!("lambda nao foi reconhecida"),
        }
    }
}

impl Parser {
    pub fn novo(tokens: Vec<TokenComPosicao>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn atual(&self) -> &Token {
        &self.tokens[self.pos].token
    }

    fn linha_atual(&self) -> usize {
        self.tokens[self.pos].linha
    }

    fn contexto_atual(&self) -> String {
        self.tokens[self.pos].contexto.clone()
    }

    fn avancar(&mut self) {
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn pular_novas_linhas(&mut self) {
        while self.atual() == &Token::NovaLinha {
            self.avancar();
        }
    }

    fn consumir(&mut self, esperado: &Token) -> Result<(), String> {
        if self.atual() == esperado {
            self.avancar();
            Ok(())
        } else {
            Err(format!(
                "Linha {}: esperava {:?}, mas encontrei {:?}",
                self.linha_atual(),
                esperado,
                self.atual()
            ))
        }
    }

    fn consumir_identificador(&mut self) -> Result<String, String> {
        match self.atual().clone() {
            Token::Identificador(nome) => {
                self.avancar();
                Ok(nome)
            }
            // Permite algumas palavras-chave curtas como nomes de variaveis
            // (util em capturar (e) { } onde 'e' e palavra-chave de 'e' logico)
            Token::E => {
                self.avancar();
                Ok("e".to_string())
            }
            Token::Ou => {
                self.avancar();
                Ok("ou".to_string())
            }
            Token::Em => {
                self.avancar();
                Ok("em".to_string())
            }
            _ => Err(format!(
                "Linha {}: esperava um nome, mas encontrei {:?}",
                self.linha_atual(),
                self.atual()
            )),
        }
    }

    fn pular_terminadores(&mut self) {
        while matches!(self.atual(), Token::NovaLinha | Token::PontoEVirgula) {
            self.avancar();
        }
    }

    fn proximo_token(&self) -> &Token {
        if self.pos + 1 < self.tokens.len() {
            &self.tokens[self.pos + 1].token
        } else {
            &Token::FimDeArquivo
        }
    }

    // -- Ponto de entrada -----------------------------------------------------

    pub fn parsear(&mut self) -> Result<Programa, String> {
        let mut instrucoes = Vec::new();
        self.pular_novas_linhas();
        while self.atual() != &Token::FimDeArquivo {
            let instr = self.parsear_instrucao()?;
            instrucoes.push(instr);
            self.pular_terminadores();
        }
        Ok(instrucoes)
    }

    fn parsear_bloco(&mut self) -> Result<Vec<Instrucao>, String> {
        self.consumir(&Token::ChaveEsq)?;
        self.pular_terminadores();
        let mut instrucoes = Vec::new();
        while self.atual() != &Token::ChaveDir && self.atual() != &Token::FimDeArquivo {
            instrucoes.push(self.parsear_instrucao()?);
            self.pular_terminadores();
        }
        self.consumir(&Token::ChaveDir)?;
        Ok(instrucoes)
    }

    fn parsear_instrucao(&mut self) -> Result<Instrucao, String> {
        let linha = self.linha_atual();
        let contexto = self.contexto_atual();
        let instrucao = self.parsear_instrucao_sem_posicao()?;
        Ok(Instrucao::Localizada {
            linha,
            contexto,
            instrucao: Box::new(instrucao),
        })
    }

    fn parsear_instrucao_sem_posicao(&mut self) -> Result<Instrucao, String> {
        match self.atual().clone() {
            Token::Var => self.parsear_declaracao_var(),
            Token::Se => self.parsear_se(),
            Token::Enquanto => self.parsear_enquanto(),
            Token::Para => self.parsear_para(),
            Token::Escolher => self.parsear_escolher(),
            Token::Funcao => self.parsear_funcao(),
            Token::Retornar => self.parsear_retornar(),
            Token::Imprimir => self.parsear_imprimir(),
            Token::Tentar => self.parsear_tentar(),
            Token::Lancar => self.parsear_lancar(),
            Token::Importar => self.parsear_importar(),
            Token::Incluir => self.parsear_incluir(false),
            Token::Requerer => self.parsear_incluir(true),
            Token::Pare => {
                self.avancar();
                Ok(Instrucao::Pare)
            }
            Token::Continue => {
                self.avancar();
                Ok(Instrucao::Continue)
            }
            _ => {
                let expr = self.parsear_expressao()?;
                Ok(Instrucao::Expressao(expr))
            }
        }
    }

    fn parsear_declaracao_var(&mut self) -> Result<Instrucao, String> {
        self.avancar();
        let nome = self.consumir_identificador()?;
        let valor = if self.atual() == &Token::Atribuicao {
            self.avancar();
            Some(self.parsear_expressao()?)
        } else {
            None
        };
        Ok(Instrucao::DeclararVar { nome, valor })
    }

    fn parsear_se(&mut self) -> Result<Instrucao, String> {
        self.avancar();
        let condicao = self.parsear_expressao()?;
        self.pular_novas_linhas();
        let entao = self.parsear_bloco()?;
        self.pular_terminadores();
        let senao = if self.atual() == &Token::Senao {
            self.avancar();
            self.pular_novas_linhas();
            if self.atual() == &Token::Se {
                Some(vec![self.parsear_se()?])
            } else {
                Some(self.parsear_bloco()?)
            }
        } else {
            None
        };
        Ok(Instrucao::Se {
            condicao,
            entao,
            senao,
        })
    }

    fn parsear_enquanto(&mut self) -> Result<Instrucao, String> {
        self.avancar();
        let condicao = self.parsear_expressao()?;
        self.pular_novas_linhas();
        let corpo = self.parsear_bloco()?;
        Ok(Instrucao::Enquanto { condicao, corpo })
    }

    fn parsear_para(&mut self) -> Result<Instrucao, String> {
        self.avancar();
        let variavel = self.consumir_identificador()?;

        if self.atual() == &Token::Em {
            // para x em lista
            self.avancar();
            let iteravel = self.parsear_expressao()?;
            self.pular_novas_linhas();
            let corpo = self.parsear_bloco()?;
            return Ok(Instrucao::Para {
                variavel,
                iteravel,
                corpo,
            });
        }

        if self.atual() == &Token::De {
            // para i de inicio ate fim [passo n]
            self.avancar();
            let inicio = self.parsear_expressao()?;
            self.consumir(&Token::Ate)?;
            let fim = self.parsear_expressao()?;
            let passo = if self.atual() == &Token::Passo {
                self.avancar();
                Some(self.parsear_expressao()?)
            } else {
                None
            };
            self.pular_novas_linhas();
            let corpo = self.parsear_bloco()?;
            return Ok(Instrucao::ParaIntervalo {
                variavel,
                inicio,
                fim,
                passo,
                corpo,
            });
        }

        Err(format!(
            "Linha {}: 'para' espera 'em' (lista) ou 'de' (intervalo)",
            self.linha_atual()
        ))
    }

    fn parsear_parametros(&mut self) -> Result<Vec<Parametro>, String> {
        let mut parametros = Vec::new();
        let mut teve_variadic = false;
        while self.atual() != &Token::ParenDir {
            if teve_variadic {
                return Err(format!(
                    "Linha {}: parametro variadic '...' deve ser o ultimo",
                    self.linha_atual()
                ));
            }
            let variadic = if self.atual() == &Token::PontoPontoPonto {
                self.avancar();
                teve_variadic = true;
                true
            } else {
                false
            };
            let nome = self.consumir_identificador()?;
            let padrao = if !variadic && self.atual() == &Token::Atribuicao {
                self.avancar();
                Some(self.parsear_expressao()?)
            } else {
                None
            };
            parametros.push(Parametro {
                nome,
                padrao,
                variadic,
            });
            if self.atual() == &Token::Virgula {
                self.avancar();
            }
        }
        Ok(parametros)
    }

    fn parsear_funcao(&mut self) -> Result<Instrucao, String> {
        self.avancar();
        let nome = self.consumir_identificador()?;
        self.consumir(&Token::ParenEsq)?;
        let parametros = self.parsear_parametros()?;
        self.consumir(&Token::ParenDir)?;
        self.pular_novas_linhas();
        let corpo = self.parsear_bloco()?;
        Ok(Instrucao::Funcao {
            nome,
            parametros,
            corpo,
        })
    }

    fn parsear_escolher(&mut self) -> Result<Instrucao, String> {
        self.avancar(); // consume 'escolher'
        let expr = self.parsear_expressao()?;
        self.pular_novas_linhas();
        self.consumir(&Token::ChaveEsq)?;
        self.pular_terminadores();

        let mut casos: Vec<(Vec<Expressao>, Vec<Instrucao>)> = Vec::new();
        let mut padrao: Option<Vec<Instrucao>> = None;

        while self.atual() != &Token::ChaveDir && self.atual() != &Token::FimDeArquivo {
            match self.atual().clone() {
                Token::Caso => {
                    self.avancar();
                    let mut valores = vec![self.parsear_expressao()?];
                    while self.atual() == &Token::Virgula {
                        self.avancar();
                        valores.push(self.parsear_expressao()?);
                    }
                    self.pular_novas_linhas();
                    let bloco = self.parsear_bloco()?;
                    casos.push((valores, bloco));
                }
                Token::Padrao => {
                    self.avancar();
                    self.pular_novas_linhas();
                    padrao = Some(self.parsear_bloco()?);
                }
                t => {
                    return Err(format!(
                        "Linha {}: esperava 'caso' ou 'padrao' em 'escolher', encontrei {:?}",
                        self.linha_atual(),
                        t
                    ))
                }
            }
            self.pular_terminadores();
        }
        self.consumir(&Token::ChaveDir)?;
        Ok(Instrucao::Escolher {
            expr,
            casos,
            padrao,
        })
    }

    fn parsear_retornar(&mut self) -> Result<Instrucao, String> {
        self.avancar();
        if matches!(
            self.atual(),
            Token::NovaLinha | Token::PontoEVirgula | Token::FimDeArquivo | Token::ChaveDir
        ) {
            Ok(Instrucao::Retornar(None))
        } else {
            Ok(Instrucao::Retornar(Some(self.parsear_expressao()?)))
        }
    }

    fn parsear_imprimir(&mut self) -> Result<Instrucao, String> {
        self.avancar();
        self.consumir(&Token::ParenEsq)?;
        let mut args = Vec::new();
        while self.atual() != &Token::ParenDir {
            args.push(self.parsear_expressao()?);
            if self.atual() == &Token::Virgula {
                self.avancar();
            }
        }
        self.consumir(&Token::ParenDir)?;
        Ok(Instrucao::Imprimir(args))
    }

    fn parsear_tentar(&mut self) -> Result<Instrucao, String> {
        self.avancar();
        self.pular_novas_linhas();
        let corpo = self.parsear_bloco()?;
        self.pular_terminadores();

        let capturar = if self.atual() == &Token::Capturar {
            self.avancar();
            self.consumir(&Token::ParenEsq)?;
            let nome_erro = self.consumir_identificador()?;
            self.consumir(&Token::ParenDir)?;
            self.pular_novas_linhas();
            let bloco = self.parsear_bloco()?;
            Some((nome_erro, bloco))
        } else {
            None
        };

        self.pular_terminadores();
        let finalmente = if self.atual() == &Token::Finalmente {
            self.avancar();
            self.pular_novas_linhas();
            Some(self.parsear_bloco()?)
        } else {
            None
        };

        Ok(Instrucao::Tentar {
            corpo,
            capturar,
            finalmente,
        })
    }

    fn parsear_lancar(&mut self) -> Result<Instrucao, String> {
        self.avancar();
        let expr = self.parsear_expressao()?;
        Ok(Instrucao::Lancar(expr))
    }

    fn parsear_importar(&mut self) -> Result<Instrucao, String> {
        self.avancar();
        if let Token::Texto(caminho) = self.atual().clone() {
            self.avancar();
            let alias = if self.atual() == &Token::Como {
                self.avancar();
                Some(self.consumir_identificador()?)
            } else {
                None
            };
            Ok(Instrucao::Importar { caminho, alias })
        } else {
            Err(format!(
                "Linha {}: importar espera um texto com o caminho do arquivo",
                self.linha_atual()
            ))
        }
    }

    fn parsear_incluir(&mut self, obrigatorio: bool) -> Result<Instrucao, String> {
        self.avancar();
        if let Token::Texto(caminho) = self.atual().clone() {
            self.avancar();
            Ok(Instrucao::Incluir {
                caminho,
                obrigatorio,
            })
        } else {
            Err(format!(
                "Linha {}: incluir/requerer espera um texto com o caminho do arquivo",
                self.linha_atual()
            ))
        }
    }

    // -- Expressoes (com precedencia) -----------------------------------------

    fn parsear_expressao(&mut self) -> Result<Expressao, String> {
        self.parsear_atribuicao()
    }

    fn parsear_atribuicao(&mut self) -> Result<Expressao, String> {
        let expr = self.parsear_ou()?;

        // Operadores compostos: +=, -=, *=, /=, %=
        let op_composto: Option<OpBinario> = match self.atual() {
            Token::MaisIgual => Some(OpBinario::Soma),
            Token::MenosIgual => Some(OpBinario::Subtracao),
            Token::EstrelaIgual => Some(OpBinario::Multiplicacao),
            Token::BarraIgual => Some(OpBinario::Divisao),
            Token::PercentualIgual => Some(OpBinario::Modulo),
            _ => None,
        };
        if let Some(op) = op_composto {
            self.avancar();
            let valor = self.parsear_atribuicao()?;
            return match expr {
                Expressao::Variavel(nome) => Ok(Expressao::Atribuicao {
                    nome: nome.clone(),
                    valor: Box::new(Expressao::BinOp {
                        esq: Box::new(Expressao::Variavel(nome)),
                        op,
                        dir: Box::new(valor),
                    }),
                }),
                Expressao::Acesso { objeto, indice } => {
                    let lhs = Expressao::Acesso {
                        objeto: objeto.clone(),
                        indice: indice.clone(),
                    };
                    Ok(Expressao::AtribuicaoIndexada {
                        objeto,
                        indice,
                        valor: Box::new(Expressao::BinOp {
                            esq: Box::new(lhs),
                            op,
                            dir: Box::new(valor),
                        }),
                    })
                }
                _ => Err(format!(
                    "Linha {}: lado esquerdo invalido para operador composto",
                    self.linha_atual()
                )),
            };
        }

        // Atribuicao simples: =
        if self.atual() == &Token::Atribuicao {
            self.avancar();
            let valor = self.parsear_atribuicao()?;
            return match expr {
                Expressao::Variavel(nome) => Ok(Expressao::Atribuicao {
                    nome,
                    valor: Box::new(valor),
                }),
                Expressao::Acesso { objeto, indice } => Ok(Expressao::AtribuicaoIndexada {
                    objeto,
                    indice,
                    valor: Box::new(valor),
                }),
                _ => Err(format!(
                    "Linha {}: lado esquerdo da atribuicao invalido",
                    self.linha_atual()
                )),
            };
        }

        Ok(expr)
    }

    fn parsear_ou(&mut self) -> Result<Expressao, String> {
        let mut esq = self.parsear_e()?;
        loop {
            if self.atual() == &Token::Ou {
                self.avancar();
                let dir = self.parsear_e()?;
                esq = Expressao::BinOp {
                    esq: Box::new(esq),
                    op: OpBinario::Ou,
                    dir: Box::new(dir),
                };
            } else if self.atual() == &Token::NullCoalescente {
                self.avancar();
                let dir = self.parsear_e()?;
                esq = Expressao::NullCoalescente {
                    esq: Box::new(esq),
                    dir: Box::new(dir),
                };
            } else {
                break;
            }
        }
        Ok(esq)
    }

    fn parsear_e(&mut self) -> Result<Expressao, String> {
        let mut esq = self.parsear_igualdade()?;
        while self.atual() == &Token::E {
            self.avancar();
            let dir = self.parsear_igualdade()?;
            esq = Expressao::BinOp {
                esq: Box::new(esq),
                op: OpBinario::E,
                dir: Box::new(dir),
            };
        }
        Ok(esq)
    }

    fn parsear_igualdade(&mut self) -> Result<Expressao, String> {
        let mut esq = self.parsear_comparacao()?;
        loop {
            let op = match self.atual() {
                Token::Igual => OpBinario::Igual,
                Token::DiferenteDe => OpBinario::DiferenteDe,
                Token::Em => OpBinario::Em,
                Token::Nao if self.proximo_token() == &Token::Em => OpBinario::NaoEm,
                _ => break,
            };
            // 'nao em' consome dois tokens
            if matches!(op, OpBinario::NaoEm) {
                self.avancar(); // consome 'nao'
            }
            self.avancar(); // consome 'em' ou o operador unico
            let dir = self.parsear_comparacao()?;
            esq = Expressao::BinOp {
                esq: Box::new(esq),
                op,
                dir: Box::new(dir),
            };
        }
        Ok(esq)
    }

    fn parsear_comparacao(&mut self) -> Result<Expressao, String> {
        let mut esq = self.parsear_soma()?;
        loop {
            let op = match self.atual() {
                Token::MenorQue => OpBinario::MenorQue,
                Token::MaiorQue => OpBinario::MaiorQue,
                Token::MenorOuIgual => OpBinario::MenorOuIgual,
                Token::MaiorOuIgual => OpBinario::MaiorOuIgual,
                _ => break,
            };
            self.avancar();
            let dir = self.parsear_soma()?;
            esq = Expressao::BinOp {
                esq: Box::new(esq),
                op,
                dir: Box::new(dir),
            };
        }
        Ok(esq)
    }

    fn parsear_soma(&mut self) -> Result<Expressao, String> {
        let mut esq = self.parsear_multiplicacao()?;
        loop {
            let op = match self.atual() {
                Token::Mais => OpBinario::Soma,
                Token::Menos => OpBinario::Subtracao,
                _ => break,
            };
            self.avancar();
            let dir = self.parsear_multiplicacao()?;
            esq = Expressao::BinOp {
                esq: Box::new(esq),
                op,
                dir: Box::new(dir),
            };
        }
        Ok(esq)
    }

    fn parsear_multiplicacao(&mut self) -> Result<Expressao, String> {
        let mut esq = self.parsear_unario()?;
        loop {
            let op = match self.atual() {
                Token::Estrela => OpBinario::Multiplicacao,
                Token::Barra => OpBinario::Divisao,
                Token::BarraBarra => OpBinario::DivisaoInteira,
                Token::Percentual => OpBinario::Modulo,
                _ => break,
            };
            self.avancar();
            let dir = self.parsear_unario()?;
            esq = Expressao::BinOp {
                esq: Box::new(esq),
                op,
                dir: Box::new(dir),
            };
        }
        Ok(esq)
    }

    fn parsear_unario(&mut self) -> Result<Expressao, String> {
        match self.atual().clone() {
            Token::Menos => {
                self.avancar();
                let expr = self.parsear_chamada()?;
                Ok(Expressao::UnOp {
                    op: OpUnario::Negativo,
                    expr: Box::new(expr),
                })
            }
            Token::Nao => {
                self.avancar();
                let expr = self.parsear_chamada()?;
                Ok(Expressao::UnOp {
                    op: OpUnario::Nao,
                    expr: Box::new(expr),
                })
            }
            _ => self.parsear_chamada(),
        }
    }

    fn parsear_chamada(&mut self) -> Result<Expressao, String> {
        let mut expr = self.parsear_primario()?;
        loop {
            match self.atual() {
                Token::ParenEsq => {
                    self.avancar();
                    let mut args = Vec::new();
                    self.pular_novas_linhas();
                    while self.atual() != &Token::ParenDir {
                        args.push(self.parsear_expressao()?);
                        self.pular_novas_linhas();
                        if self.atual() == &Token::Virgula {
                            self.avancar();
                            self.pular_novas_linhas();
                        }
                    }
                    self.consumir(&Token::ParenDir)?;
                    if let Expressao::Variavel(nome) = expr {
                        expr = Expressao::ChamadaFuncao { nome, args };
                    } else {
                        expr = Expressao::Chamada {
                            funcao: Box::new(expr),
                            args,
                        };
                    }
                }
                Token::ColcheteEsq => {
                    self.avancar();
                    let indice = self.parsear_expressao()?;
                    self.consumir(&Token::ColcheteDir)?;
                    expr = Expressao::Acesso {
                        objeto: Box::new(expr),
                        indice: Box::new(indice),
                    };
                }
                Token::Ponto => {
                    self.avancar();
                    let campo = self.consumir_identificador()?;
                    expr = Expressao::Acesso {
                        objeto: Box::new(expr),
                        indice: Box::new(Expressao::Texto(campo)),
                    };
                }
                Token::PontoOpcional => {
                    // `a?.campo` — acesso com verificação de nulo
                    self.avancar();
                    let campo = self.consumir_identificador()?;
                    expr = Expressao::AcessoOpcional {
                        objeto: Box::new(expr),
                        chave: campo,
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parsear_primario(&mut self) -> Result<Expressao, String> {
        match self.atual().clone() {
            Token::Inteiro(n) => {
                self.avancar();
                Ok(Expressao::Inteiro(n))
            }
            Token::Numero(n) => {
                self.avancar();
                Ok(Expressao::Numero(n))
            }
            Token::Texto(s) => {
                self.avancar();
                // Interpolacao de strings: "Ola, {nome}!"
                if s.contains('{') {
                    parsear_interpolacao(s)
                } else {
                    Ok(Expressao::Texto(s))
                }
            }
            Token::Verdadeiro => {
                self.avancar();
                Ok(Expressao::Booleano(true))
            }
            Token::Falso => {
                self.avancar();
                Ok(Expressao::Booleano(false))
            }
            Token::Nulo => {
                self.avancar();
                Ok(Expressao::Nulo)
            }
            Token::Funcao => self.parsear_funcao_anonima(),
            Token::Identificador(nome) => {
                self.avancar();
                // Funcao seta: `x => expressao`
                if self.atual() == &Token::Seta {
                    self.avancar();
                    let corpo = self.parsear_expressao()?;
                    return Ok(Expressao::FuncaoSeta {
                        parametros: vec![nome],
                        corpo: Box::new(corpo),
                    });
                }
                Ok(Expressao::Variavel(nome))
            }
            // Permite palavras-chave ambiguas como variaveis em posicao primaria
            Token::E => {
                self.avancar();
                Ok(Expressao::Variavel("e".to_string()))
            }
            Token::Ou => {
                self.avancar();
                Ok(Expressao::Variavel("ou".to_string()))
            }
            Token::ParenEsq => {
                // Tenta detectar funcao seta multi-param: `(a, b) => expr`
                if let Some(params) = self.tentar_params_seta() {
                    self.consumir(&Token::Seta)?;
                    let corpo = self.parsear_expressao()?;
                    return Ok(Expressao::FuncaoSeta {
                        parametros: params,
                        corpo: Box::new(corpo),
                    });
                }
                self.avancar();
                let expr = self.parsear_expressao()?;
                self.consumir(&Token::ParenDir)?;
                Ok(expr)
            }
            Token::ColcheteEsq => {
                self.avancar();
                let mut elementos = Vec::new();
                self.pular_novas_linhas();
                while self.atual() != &Token::ColcheteDir {
                    elementos.push(self.parsear_expressao()?);
                    self.pular_novas_linhas();
                    if self.atual() == &Token::Virgula {
                        self.avancar();
                        self.pular_novas_linhas();
                    }
                }
                self.consumir(&Token::ColcheteDir)?;
                Ok(Expressao::Lista(elementos))
            }
            Token::ChaveEsq => {
                self.avancar();
                let mut pares = Vec::new();
                self.pular_novas_linhas();
                while self.atual() != &Token::ChaveDir && self.atual() != &Token::FimDeArquivo {
                    let chave = match self.atual().clone() {
                        Token::Texto(s) => {
                            self.avancar();
                            s
                        }
                        Token::Identificador(s) => {
                            self.avancar();
                            s
                        }
                        t => {
                            return Err(format!(
                                "Linha {}: chave de mapa invalida: {:?}",
                                self.linha_atual(),
                                t
                            ))
                        }
                    };
                    self.consumir(&Token::DoisPontos)?;
                    let valor = self.parsear_expressao()?;
                    pares.push((chave, valor));
                    self.pular_novas_linhas();
                    if self.atual() == &Token::Virgula {
                        self.avancar();
                        self.pular_novas_linhas();
                    }
                }
                self.consumir(&Token::ChaveDir)?;
                Ok(Expressao::Mapa(pares))
            }
            t => Err(format!(
                "Linha {}: expressao inesperada: {:?}",
                self.linha_atual(),
                t
            )),
        }
    }

    /// Lookahead para detectar `(a, b) => ...` sem consumir tokens se falhar.
    fn tentar_params_seta(&mut self) -> Option<Vec<String>> {
        let salvo = self.pos;
        // Deve começar com `(`
        if self.atual() != &Token::ParenEsq {
            return None;
        }
        self.pos += 1;
        let mut params = Vec::new();
        loop {
            match self.atual() {
                Token::ParenDir => {
                    self.pos += 1;
                    break;
                }
                Token::Identificador(nome) => {
                    let n = nome.clone();
                    self.pos += 1;
                    params.push(n);
                    match self.atual() {
                        Token::Virgula => {
                            self.pos += 1;
                        }
                        Token::ParenDir => {
                            self.pos += 1;
                            break;
                        }
                        _ => {
                            self.pos = salvo;
                            return None;
                        }
                    }
                }
                _ => {
                    self.pos = salvo;
                    return None;
                }
            }
        }
        // Deve ser seguido de `=>`
        if self.atual() != &Token::Seta {
            self.pos = salvo;
            return None;
        }
        Some(params)
    }

    fn parsear_funcao_anonima(&mut self) -> Result<Expressao, String> {
        self.avancar();
        self.consumir(&Token::ParenEsq)?;
        let parametros = self.parsear_parametros()?;
        self.consumir(&Token::ParenDir)?;
        self.pular_novas_linhas();
        let corpo = self.parsear_bloco()?;
        Ok(Expressao::FuncaoAnonima { parametros, corpo })
    }
}

// -- Interpolacao de strings ---------------------------------------------------

/// Transforma "Ola, {nome}!" em: "Ola, " + texto(nome) + "!"
fn parsear_interpolacao(s: String) -> Result<Expressao, String> {
    let mut partes: Vec<Expressao> = Vec::new();
    let mut texto_atual = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '{' if chars.get(i + 1) == Some(&'{') => {
                texto_atual.push('{');
                i += 2;
            }
            '{' => {
                if !texto_atual.is_empty() {
                    partes.push(Expressao::Texto(texto_atual.clone()));
                    texto_atual.clear();
                }
                i += 1;
                let mut expr_str = String::new();
                let mut depth = 1usize;
                while i < chars.len() {
                    match chars[i] {
                        '{' => {
                            depth += 1;
                            expr_str.push(chars[i]);
                            i += 1;
                        }
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                i += 1;
                                break;
                            }
                            expr_str.push(chars[i]);
                            i += 1;
                        }
                        c => {
                            expr_str.push(c);
                            i += 1;
                        }
                    }
                }
                let mut lex = Lexer::novo(&expr_str);
                let tokens = lex.tokenizar()?;
                let mut p = Parser::novo(tokens);
                let expr = p.parsear_expressao()?;
                partes.push(Expressao::ChamadaFuncao {
                    nome: "texto".to_string(),
                    args: vec![expr],
                });
            }
            '}' if chars.get(i + 1) == Some(&'}') => {
                texto_atual.push('}');
                i += 2;
            }
            c => {
                texto_atual.push(c);
                i += 1;
            }
        }
    }

    if !texto_atual.is_empty() {
        partes.push(Expressao::Texto(texto_atual));
    }

    if partes.is_empty() {
        return Ok(Expressao::Texto(String::new()));
    }
    if partes.len() == 1 {
        return Ok(partes.remove(0));
    }

    let mut resultado = partes.remove(0);
    for parte in partes {
        resultado = Expressao::BinOp {
            esq: Box::new(resultado),
            op: OpBinario::Soma,
            dir: Box::new(parte),
        };
    }
    Ok(resultado)
}
