pub fn init_logger() {
    let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
    env_logger::Builder::from_env(env)
        .format(|buf, record| {
            use std::io::Write;
            let ts = chrono::Local::now().format("[%H:%M:%S]");
            writeln!(buf, "{} [{}] {}", ts, record.level(), record.args())
        })
        .init();
}