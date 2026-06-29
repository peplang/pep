fn main() {
    #[cfg(target_os = "windows")]
    {
        let mut res = winresource::WindowsResource::new();

        // Ícone do executável (aparece no Explorer e na barra de tarefas)
        res.set_icon("assets/pep.ico");

        // Propriedades visíveis em Propriedades → Detalhes no Explorer
        res.set("FileDescription", "PEP - Programar em Portugues");
        res.set("ProductName", "PEP");
        res.set("FileVersion", "0.6.0.0");
        res.set("ProductVersion", "0.6.0.0");
        res.set("LegalCopyright", "Copyright 2026 PEP");
        res.set("CompanyName", "PEP Open Source");
        res.set("OriginalFilename", "pep.exe");
        res.set("InternalName", "pep");

        res.compile().expect("Falha ao compilar recursos do Windows");
    }
}
