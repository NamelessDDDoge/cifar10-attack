from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

MIXED_ARGS = {
    "--cnn-count 2",
    "--vit-surrogates vit_hf_nateraw,vit_timm_edadaltocg",
    "--robust-surrogates robust_engstrom,robust_rade_r18_extra,robust_rebuffi_70_16_cutmix_extra,robust_xcit_s12",
}


def test_mixed_non_pgd_attack_scripts_keep_same_surrogate_pool():
    expected = {
        "attack_mixed_pgn_smoke.sh": "code/attack/run_pgn.py",
        "attack_mixed_bsr_smoke.sh": "code/attack/run_bsr.py",
        "attack_mixed_ilpd_smoke.sh": "code/attack/run_ilpd.py",
        "attack_mixed_awt_smoke.sh": "code/attack/run_awt.py",
    }

    for script_name, runner in expected.items():
        text = (SCRIPTS / script_name).read_text(encoding="utf-8")
        assert runner in text
        assert "--out-dir results/adv_" in text
        assert "--max-images 8" in text
        assert "--batch-size 1" in text
        for arg in MIXED_ARGS:
            assert arg in text
