/// 位相（Phase）を Z/NZ 巡回群として定義
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phase(pub u32);

#[derive(Debug, Clone, Copy)]
pub enum SyncRequirement {
    None,
    /// cp.async.wait_group<N> に相当
    WaitAsyncLoad { stages_behind: u32 },
    /// __syncthreads() 相当
    Barrier,
}

#[derive(Debug, Clone)]
pub struct PhaseTransition {
    pub from: Phase,
    pub to: Phase,
    pub num_stages: u32,
}

impl PhaseTransition {
    pub fn new(from: u32, to: u32, num_stages: u32) -> Self {
        Self {
            from: Phase(from % num_stages),
            to: Phase(to % num_stages),
            num_stages,
        }
    }

    /// 次のステージへ進むための同期命令を算出
    /// ステージ間の距離に基づいて同期の重みを決定
    pub fn required_sync(&self) -> SyncRequirement {
        // 例: Nステージパイプラインの場合、
        // ロードの完了待機は (num_stages - 2) 個前を指定するのが一般的
        // (最新のcommitから数えていくつ前のグループが完了しているべきか)
        if self.num_stages > 1 {
            SyncRequirement::WaitAsyncLoad {
                stages_behind: (self.num_stages.saturating_sub(2)),
            }
        } else {
            SyncRequirement::Barrier
        }
    }
}
