use std::error::Error;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt::SubscriberBuilder;

#[inline]
pub fn init_logger(level: Option<LevelFilter>) -> Result<(), Box<dyn Error + Send + Sync>> {
    init_logger_with_customization(|builder| {
        builder.with_max_level(level.unwrap_or_else(|| LevelFilter::WARN))
    })
}

pub fn init_logger_with_customization(
    f: impl FnOnce(SubscriberBuilder) -> SubscriberBuilder,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    f(tracing_subscriber::fmt()
        .with_line_number(true)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE))
    .try_init()
}
