# PEP — Programar em Português

O **PEP** é uma linguagem de programação interpretada, de tipagem dinâmica, desenvolvida em **Rust**. O objetivo principal é unir a clareza e a expressividade do idioma português a recursos modernos de engenharia de software, incluindo suporte nativo para desenvolvimento web, WebSocket, banco de dados, computação numérica (tensores) e inteligência artificial (autodiff).

---

## 🚀 Recursos Principais

* **Sintaxe limpa e intuitiva:** Instruções encerradas por quebras de linha e blocos delimitados por chaves `{}`.
* **Web Integrada:** Servidor HTTP nativo com roteamento baseado em arquivos, suporte a templates `.phtml` (HTML misturado com código PEP) e manipulação automática de contexto (Cookies, Sessões, Uploads, JSON).
* **WebSocket Nativo:** Implementação do protocolo RFC 6455 sem dependências externas, operando com gerenciamento automático de threads.
* **Mapeamento de Memória e Bytes:** Computação de baixo nível para manipulação eficiente de arquivos binários e leitura de pesos através de `mmap`.
* **Computação Científica & IA:** Motor interno de diferenciação automática reversa (*backpropagation*), suporte a tensores (com operações executáveis em CPU ou aceleradas por GPU/CUDA/Metal) e quantização em `int8` e `float16`.

---

## 🛠️ Instalação e Execução

Como o interpretador é construído sobre o ecossistema Rust, certifique-se de ter o **Cargo** instalado e execute a compilação na raiz do projeto:

```bash
cargo build --release