use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use tracing_appender::rolling;

pub fn init_logger(verbosity: &str) -> anyhow::Result<tracing_appender::non_blocking::WorkerGuard> {
    let log_level = match verbosity {
        "silent" => "off",
        "info" => "info",
        "debug" => "debug",
        "trace" => "trace",
        _ => "info",
    };

    let file_appender = rolling::daily("./logs", "turbo-quant.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let filter = EnvFilter::new(log_level);

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(std::io::stdout).pretty())
        .with(fmt::layer().with_writer(non_blocking).json())
        .init();

    Ok(guard)
}
