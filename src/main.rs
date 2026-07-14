//! エントリポイント。
//!
//! TOML 設定ファイルを読み込み、サブコマンドに応じて一括シミュレーション（run）
//! またはアニメーション表示（animate）を実行する。

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

mod config;
#[cfg(feature = "gpu")]
mod gpu;
mod model;
mod renderer;
mod runner;
mod simulation;

/// 周期チャネル内を進む棒状粒子のブラウン運動シミュレータ
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// 設定ファイル（TOML）のパス
    #[arg(short, long, default_value = "config.toml", global = true)]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// 設定された全ケースのシミュレーションを実行する（既定）
    Run,
    /// 1粒子のアニメーションを GUI で表示する
    Animate,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = config::Config::load(&cli.config)?;

    match cli.command.unwrap_or(Command::Run) {
        Command::Run => runner::run_all(&config, &cli.config),
        Command::Animate => renderer::run_animation(&config),
    }
}
