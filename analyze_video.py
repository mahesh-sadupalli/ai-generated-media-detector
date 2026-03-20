"""Analyze a video file with the full artifact detection pipeline.

Usage:
    python analyze_video.py <video_path> [--max-frames N] [--save-report DIR]
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

from src.artifact_detectors.combined_artifact_classifier import ArtifactClassifier
from src.utils.simple_face_detection import SimpleFaceDetector


def analyze_video(video_path: str, max_frames: int = 15,
                  save_report: str | None = None) -> dict:
    """Run the full detection pipeline on a video and print results.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    max_frames : int
        Maximum number of frames to sample.
    save_report : str or None
        If provided, save visualisation charts to this directory.

    Returns
    -------
    dict
        Full analysis results including per-frame data.
    """
    path = Path(video_path)
    if not path.exists():
        print(f"ERROR: File not found: {video_path}")
        return {}

    print(f"{'=' * 70}")
    print(f"  AI-GENERATED MEDIA DETECTOR — VIDEO ANALYSIS")
    print(f"  File: {path.name}")
    print(f"{'=' * 70}\n")

    # --- Video info ---
    cap = cv2.VideoCapture(str(path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        print(f"  Resolution:  {width}x{height}")
        print(f"  FPS:         {fps:.1f}")
        print(f"  Duration:    {duration:.1f}s ({total_frames} frames)")
        print(f"  Sampling:    {max_frames} frames\n")
    finally:
        cap.release()

    # --- Extract faces ---
    print("  [1/3] Extracting faces ...")
    face_extractor = SimpleFaceDetector()
    faces = face_extractor.extract_faces_from_video(str(path), max_frames=max_frames)

    if not faces:
        print("  ERROR: No faces detected in video.\n")
        return {}

    print(f"         Found {len(faces)} face(s)\n")

    # --- Run classifier on each face ---
    print("  [2/3] Running artifact analysis ...\n")
    classifier = ArtifactClassifier()
    frame_results = []

    for i, face in enumerate(faces):
        result = classifier.analyze_image(face)
        result['_face'] = face  # keep for visualisation
        frame_results.append(result)

        scores = result['scores']
        print(f"  Frame {i + 1:>2}/{len(faces)}  |  "
              f"Prediction: {result['prediction']:<22s}  |  "
              f"GAN: {scores['gan_overall']:.3f}  "
              f"Diff: {scores['diffusion']:.3f}->{scores['diffusion_adjusted']:.3f}  "
              f"Smooth: {scores['smoothing']:.3f}->{scores['smoothing_adjusted']:.3f}  "
              f"Comp: {scores['compression_level']:.2f}(x{scores['compression_attenuation']:.2f})")

    # --- Aggregate ---
    print(f"\n  [3/3] Aggregating results ...\n")
    print(f"{'=' * 70}")

    votes = {'REAL': 0, 'GAN-GENERATED': 0, 'DIFFUSION-GENERATED': 0}
    for r in frame_results:
        votes[r['prediction']] = votes.get(r['prediction'], 0) + 1

    verdict = max(votes, key=votes.get)
    avg_confidence = np.mean([r['confidence'] for r in frame_results])
    avg_gan = np.mean([r['scores']['gan_overall'] for r in frame_results])
    avg_diff = np.mean([r['scores']['diffusion'] for r in frame_results])
    avg_diff_adj = np.mean([r['scores']['diffusion_adjusted'] for r in frame_results])
    avg_smooth = np.mean([r['scores']['smoothing'] for r in frame_results])
    avg_smooth_adj = np.mean([r['scores']['smoothing_adjusted'] for r in frame_results])
    avg_texture = np.mean([r['scores']['texture'] for r in frame_results])
    avg_collapse = np.mean([r['scores']['mode_collapse'] for r in frame_results])
    avg_compression = np.mean([r['scores']['compression_level'] for r in frame_results])
    avg_attenuation = np.mean([r['scores']['compression_attenuation'] for r in frame_results])

    print(f"\n  VERDICT:  {verdict}")
    print(f"  Confidence: {avg_confidence:.3f}\n")

    print(f"  Vote breakdown:")
    for cls, count in votes.items():
        bar = '#' * count
        print(f"    {cls:<22s}  {count:>2}/{len(faces)}  {bar}")

    print(f"\n  Average scores:")
    print(f"    GAN overall:    {avg_gan:.3f}")
    print(f"    Diffusion:      {avg_diff:.3f} (raw) -> {avg_diff_adj:.3f} (adjusted)")
    print(f"    Smoothing:      {avg_smooth:.3f} (raw) -> {avg_smooth_adj:.3f} (adjusted)")
    print(f"    Texture:        {avg_texture:.3f}")
    print(f"    Mode collapse:  {avg_collapse:.3f}")
    print(f"    Compression:    {avg_compression:.3f} (attenuation: x{avg_attenuation:.3f})")

    print(f"\n{'=' * 70}\n")

    analysis = {
        'video_info': {
            'file': path.name,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'duration': duration,
            'total_frames': total_frames,
        },
        'verdict': verdict,
        'confidence': avg_confidence,
        'votes': votes,
        'faces_found': len(faces),
        'avg_scores': {
            'gan_overall': avg_gan,
            'diffusion': avg_diff,
            'diffusion_adjusted': avg_diff_adj,
            'smoothing': avg_smooth,
            'smoothing_adjusted': avg_smooth_adj,
            'texture': avg_texture,
            'mode_collapse': avg_collapse,
            'compression_level': avg_compression,
            'compression_attenuation': avg_attenuation,
        },
        'frame_results': frame_results,
    }

    if save_report:
        _generate_report(analysis, save_report)

    return analysis


def _generate_report(analysis: dict, output_dir: str) -> None:
    """Generate diagnostic visualisation charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = analysis['frame_results']
    n = len(results)
    indices = np.arange(1, n + 1)

    gan_scores = [r['scores']['gan_overall'] for r in results]
    diff_scores = [r['scores']['diffusion'] for r in results]
    smooth_scores = [r['scores']['smoothing'] for r in results]
    texture_scores = [r['scores']['texture'] for r in results]
    collapse_scores = [r['scores']['mode_collapse'] for r in results]
    predictions = [r['prediction'] for r in results]

    pred_colors = {
        'REAL': '#2ecc71',
        'GAN-GENERATED': '#e74c3c',
        'DIFFUSION-GENERATED': '#9b59b6',
    }
    frame_colors = [pred_colors.get(p, '#95a5a6') for p in predictions]

    # ─── Chart 1: Per-frame score timeline ───────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Per-Frame Detection Scores — {analysis['video_info']['file']}",
        fontsize=14, fontweight='bold',
    )

    ax1 = axes[0]
    ax1.plot(indices, gan_scores, 'o-', color='#e74c3c', label='GAN score',
             markersize=3, linewidth=1)
    ax1.plot(indices, diff_scores, 's-', color='#9b59b6', label='Diffusion score',
             markersize=3, linewidth=1)
    ax1.axhline(y=0.60, color='#e74c3c', linestyle='--', alpha=0.5,
                label='GAN threshold (0.60)')
    ax1.axhline(y=0.55, color='#9b59b6', linestyle='--', alpha=0.5,
                label='Diffusion threshold (0.55)')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('GAN vs Diffusion Scores (with decision thresholds)')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(indices, smooth_scores, 'o-', color='#3498db', label='Smoothing',
             markersize=3, linewidth=1)
    ax2.plot(indices, texture_scores, 's-', color='#f39c12', label='Texture',
             markersize=3, linewidth=1)
    ax2.plot(indices, collapse_scores, '^-', color='#1abc9c', label='Mode collapse',
             markersize=3, linewidth=1)
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Frame #')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_title('Individual Detector Scores')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / '1_score_timeline.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / '1_score_timeline.png'}")

    # ─── Chart 2: Score distributions (histograms) ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Score Distributions Across All Frames', fontsize=13,
                 fontweight='bold')

    ax1 = axes[0]
    ax1.hist(gan_scores, bins=20, alpha=0.7, color='#e74c3c', label='GAN')
    ax1.hist(diff_scores, bins=20, alpha=0.7, color='#9b59b6', label='Diffusion')
    ax1.axvline(x=0.60, color='#e74c3c', linestyle='--', linewidth=2,
                label='GAN threshold')
    ax1.axvline(x=0.55, color='#9b59b6', linestyle='--', linewidth=2,
                label='Diffusion threshold')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Frame count')
    ax1.set_title('GAN vs Diffusion Score Distribution')
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 1)

    ax2 = axes[1]
    ax2.hist(smooth_scores, bins=20, alpha=0.6, color='#3498db',
             label='Smoothing')
    ax2.hist(collapse_scores, bins=20, alpha=0.6, color='#1abc9c',
             label='Mode collapse')
    ax2.hist(texture_scores, bins=20, alpha=0.6, color='#f39c12',
             label='Texture')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Frame count')
    ax2.set_title('Sub-detector Score Distributions')
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    fig.savefig(out / '2_score_distributions.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / '2_score_distributions.png'}")

    # ─── Chart 3: Diffusion sub-detector breakdown ───────────────────
    recon_scores = [r['details']['diffusion']['reconstruction_error']
                    for r in results]
    spectral_scores = [r['details']['diffusion']['spectral_fingerprint']
                       for r in results]
    noise_scores = [r['details']['diffusion']['noise_residual']
                    for r in results]
    patch_scores = [r['details']['diffusion']['patch_consistency']
                    for r in results]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(indices, recon_scores, 'o-', label='Reconstruction error',
            markersize=3, linewidth=1, color='#e74c3c')
    ax.plot(indices, spectral_scores, 's-', label='Spectral fingerprint',
            markersize=3, linewidth=1, color='#3498db')
    ax.plot(indices, noise_scores, '^-', label='Noise residual',
            markersize=3, linewidth=1, color='#2ecc71')
    ax.plot(indices, patch_scores, 'D-', label='Patch consistency',
            markersize=3, linewidth=1, color='#f39c12')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frame #')
    ax.set_ylabel('Sub-score')
    ax.set_ylim(0, 1)
    ax.set_title('Diffusion Detector Breakdown — Which Sub-detector Is Firing?',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / '3_diffusion_breakdown.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / '3_diffusion_breakdown.png'}")

    # ─── Chart 4: Classification confusion — GAN vs Diffusion scatter ─
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(n):
        ax.scatter(gan_scores[i], diff_scores[i], c=frame_colors[i],
                   s=40, edgecolors='white', linewidth=0.5, zorder=3)

    ax.axhline(y=0.55, color='#9b59b6', linestyle='--', alpha=0.5)
    ax.axvline(x=0.60, color='#e74c3c', linestyle='--', alpha=0.5)

    # quadrant labels
    ax.text(0.30, 0.80, 'DIFFUSION-\nGENERATED', ha='center', fontsize=10,
            color='#9b59b6', alpha=0.6, fontweight='bold')
    ax.text(0.80, 0.30, 'GAN-\nGENERATED', ha='center', fontsize=10,
            color='#e74c3c', alpha=0.6, fontweight='bold')
    ax.text(0.30, 0.25, 'REAL', ha='center', fontsize=12,
            color='#2ecc71', alpha=0.6, fontweight='bold')
    ax.text(0.80, 0.80, 'BOTH\nFLAGGED', ha='center', fontsize=10,
            color='#95a5a6', alpha=0.6, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=10, label='Classified REAL'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=10, label='Classified GAN'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9b59b6',
               markersize=10, label='Classified DIFFUSION'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.set_xlabel('GAN Score', fontsize=12)
    ax.set_ylabel('Diffusion Score', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('GAN vs Diffusion Score — Per-Frame Classification',
                 fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / '4_gan_vs_diffusion_scatter.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / '4_gan_vs_diffusion_scatter.png'}")

    # ─── Chart 5: Sample faces grid with predictions ─────────────────
    sample_indices = np.linspace(0, n - 1, min(12, n), dtype=int)
    cols = 4
    rows = int(np.ceil(len(sample_indices) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    fig.suptitle('Sample Faces with Predictions', fontsize=14,
                 fontweight='bold')

    if rows == 1:
        axes = [axes]

    for idx, ax_row in enumerate(axes):
        if not hasattr(ax_row, '__iter__'):
            ax_row = [ax_row]
        for jdx, ax in enumerate(ax_row):
            flat_idx = idx * cols + jdx
            if flat_idx < len(sample_indices):
                si = sample_indices[flat_idx]
                face = results[si]['_face']
                pred = results[si]['prediction']
                conf = results[si]['confidence']

                ax.imshow(face)
                color = pred_colors.get(pred, '#95a5a6')
                ax.set_title(f"#{si + 1}: {pred}\n({conf:.3f})",
                             fontsize=9, color=color, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(out / '5_sample_faces.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out / '5_sample_faces.png'}")

    print(f"\n  All charts saved to: {out}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze video for AI artifacts')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--max-frames', type=int, default=15,
                        help='Max frames to sample (default: 15)')
    parser.add_argument('--save-report', type=str, default=None,
                        help='Directory to save visualisation charts')
    args = parser.parse_args()
    analyze_video(args.video_path, args.max_frames, args.save_report)
