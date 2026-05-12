use tracea::optimizer::{GaussianProcess, HardwareProfile};
use tracea::optimizer::benchmark::Observation;
use tracea::PipelineConfig;

#[test]
fn test_gp_interpolation_and_prediction() {
    let mut gp = GaussianProcess::new();
    // Set a very low noise_sigma so that GP behaves as a true interpolator
    gp.noise_sigma = 0.001; 

    let gpu = HardwareProfile::rtx3070();

    let m = 256;
    let n = 256;
    let k = 256;

    // Create distinct configurations as training observations
    let config1 = PipelineConfig::new(1, 16, 16, 16);
    let config2 = PipelineConfig::new(2, 32, 32, 16);
    let config3 = PipelineConfig::new(2, 64, 64, 16);

    let score1 = 15.5f32; // Simulated TFLOPS scores
    let score2 = 42.1f32;
    let score3 = 28.3f32;

    gp.observe(Observation { m, n, k, config: config1.clone(), score: score1 }, &gpu);
    gp.observe(Observation { m, n, k, config: config2.clone(), score: score2 }, &gpu);
    gp.observe(Observation { m, n, k, config: config3.clone(), score: score3 }, &gpu);

    // Verify interpolation: prediction AT training points must match scores closely
    let (pred_mean1, pred_std1) = gp.predict(m, n, k, &config1, &gpu);
    let (pred_mean2, pred_std2) = gp.predict(m, n, k, &config2, &gpu);
    let (pred_mean3, pred_std3) = gp.predict(m, n, k, &config3, &gpu);

    println!("Observation 1 Score: {}, Prediction: {} (std: {})", score1, pred_mean1, pred_std1);
    println!("Observation 2 Score: {}, Prediction: {} (std: {})", score2, pred_mean2, pred_std2);
    println!("Observation 3 Score: {}, Prediction: {} (std: {})", score3, pred_mean3, pred_std3);

    // Assert that predicted means are very close to observed scores at observation sites
    assert!((pred_mean1 - score1).abs() < 0.2, "GP interpolation failure for config1");
    assert!((pred_mean2 - score2).abs() < 0.2, "GP interpolation failure for config2");
    assert!((pred_mean3 - score3).abs() < 0.2, "GP interpolation failure for config3");

    // Prediction standard deviation (uncertainty) at exact training spots must be extremely small
    assert!(pred_std1 < 0.1, "GP uncertainty should be near-zero at observation 1");
    assert!(pred_std2 < 0.1, "GP uncertainty should be near-zero at observation 2");
    assert!(pred_std3 < 0.1, "GP uncertainty should be near-zero at observation 3");
}
