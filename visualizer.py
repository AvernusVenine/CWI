import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.gaussian_process import GaussianProcessClassifier


def visualize_3d_probability_surface(model, class_idx=0,
                                     utme_range=(450000, 550000),
                                     utmn_range=(4450000, 4550000),
                                     elevation_fixed=300,
                                     n_points=50):
    """
    Create 3D surface plot showing probability across UTME and UTMN space (elevation fixed).

    Parameters:
    -----------
    model : GaussianProcessClassifier
        Trained GP model
    class_idx : int
        Which class to visualize (0, 1, or 2)
    utme_range : tuple
        (min, max) UTME values
    utmn_range : tuple
        (min, max) UTMN values
    elevation_fixed : float
        Fixed elevation value
    n_points : int
        Grid resolution (n_points x n_points)
    """

    # Create meshgrid
    utme_vals = np.linspace(utme_range[0], utme_range[1], n_points)
    utmn_vals = np.linspace(utmn_range[0], utmn_range[1], n_points)
    UTME, UTMN = np.meshgrid(utme_vals, utmn_vals)

    # Flatten for prediction
    utme_flat = UTME.flatten()
    utmn_flat = UTMN.flatten()
    elevation_flat = np.full(len(utme_flat), elevation_fixed)

    # Create feature matrix
    X_viz = np.column_stack([utme_flat, utmn_flat, elevation_flat])

    # Get predictions
    probabilities = model.predict_proba(X_viz)
    prob_grid = probabilities[:, class_idx].reshape(UTME.shape)

    # Class labels
    class_labels = ['Cedar Valley', 'Wapsipinicon', 'None']

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(UTME, UTMN, prob_grid, cmap='viridis',
                           alpha=0.9, edgecolor='none', antialiased=True)

    # Add contour lines on bottom
    ax.contour(UTME, UTMN, prob_grid, zdir='z', offset=0,
               cmap='viridis', alpha=0.5, linewidths=1)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label(f'P({class_labels[class_idx]})', fontsize=12, fontweight='bold')

    # Labels
    ax.set_xlabel('UTME (m)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('UTMN (m)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Probability', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(f'3D Probability Surface: {class_labels[class_idx]}\n(Elevation={elevation_fixed:.1f}m)',
                 fontsize=13, fontweight='bold', pad=20)

    ax.set_zlim([0, 1])
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(f'3d_surface_{class_labels[class_idx].replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def visualize_3d_probability_volume(model, class_idx=0,
                                    utme_range=(450000, 550000),
                                    utmn_fixed=4500000,
                                    elevation_range=(200, 400),
                                    n_points=50):
    """
    Create 3D surface plot showing probability across UTME and elevation (UTMN fixed).

    Parameters:
    -----------
    model : GaussianProcessClassifier
        Trained GP model
    class_idx : int
        Which class to visualize
    utme_range : tuple
        (min, max) UTME values
    utmn_fixed : float
        Fixed UTMN value
    elevation_range : tuple
        (min, max) elevation values
    n_points : int
        Grid resolution
    """

    # Create meshgrid
    utme_vals = np.linspace(utme_range[0], utme_range[1], n_points)
    elevation_vals = np.linspace(elevation_range[0], elevation_range[1], n_points)
    UTME, ELEVATION = np.meshgrid(utme_vals, elevation_vals)

    # Flatten for prediction
    utme_flat = UTME.flatten()
    elevation_flat = ELEVATION.flatten()
    utmn_flat = np.full(len(utme_flat), utmn_fixed)

    # Create feature matrix
    X_viz = np.column_stack([utme_flat, utmn_flat, elevation_flat])

    # Get predictions
    probabilities = model.predict_proba(X_viz)
    prob_grid = probabilities[:, class_idx].reshape(UTME.shape)

    # Class labels
    class_labels = ['Cedar Valley', 'Wapsipinicon', 'None']

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(UTME, ELEVATION, prob_grid, cmap='plasma',
                           alpha=0.9, edgecolor='none', antialiased=True)

    # Add contour lines on bottom
    ax.contour(UTME, ELEVATION, prob_grid, zdir='z', offset=0,
               cmap='plasma', alpha=0.5, linewidths=1)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label(f'P({class_labels[class_idx]})', fontsize=12, fontweight='bold')

    # Labels
    ax.set_xlabel('UTME (m)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Elevation (m)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Probability', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(f'3D Probability Surface: {class_labels[class_idx]}\n(UTMN={utmn_fixed:,.0f})',
                 fontsize=13, fontweight='bold', pad=20)

    ax.set_zlim([0, 1])
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(f'3d_utme_elev_{class_labels[class_idx].replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def visualize_3d_all_classes(model, utme_range=(450000, 550000),
                             utmn_range=(4450000, 4550000),
                             elevation_fixed=300,
                             n_points=50):
    """
    Create 3D surface plots for all classes side by side.

    Parameters:
    -----------
    model : GaussianProcessClassifier
        Trained GP model
    utme_range : tuple
        (min, max) UTME values
    utmn_range : tuple
        (min, max) UTMN values
    elevation_fixed : float
        Fixed elevation value
    n_points : int
        Grid resolution
    """

    # Create meshgrid
    utme_vals = np.linspace(utme_range[0], utme_range[1], n_points)
    utmn_vals = np.linspace(utmn_range[0], utmn_range[1], n_points)
    UTME, UTMN = np.meshgrid(utme_vals, utmn_vals)

    # Flatten for prediction
    utme_flat = UTME.flatten()
    utmn_flat = UTMN.flatten()
    elevation_flat = np.full(len(utme_flat), elevation_fixed)

    # Create feature matrix
    X_viz = np.column_stack([utme_flat, utmn_flat, elevation_flat])

    # Get predictions
    probabilities = model.predict_proba(X_viz)

    # Class labels
    class_labels = ['Cedar Valley', 'Wapsipinicon', 'None']
    colormaps = ['viridis', 'plasma', 'inferno']

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))

    for idx in range(3):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

        prob_grid = probabilities[:, idx].reshape(UTME.shape)

        # Plot surface
        surf = ax.plot_surface(UTME, UTMN, prob_grid, cmap=colormaps[idx],
                               alpha=0.9, edgecolor='none', antialiased=True)

        # Add contour lines
        ax.contour(UTME, UTMN, prob_grid, zdir='z', offset=0,
                   cmap=colormaps[idx], alpha=0.4, linewidths=0.8)

        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Probability', fontsize=10)

        # Labels
        ax.set_xlabel('UTME', fontsize=10, labelpad=8)
        ax.set_ylabel('UTMN', fontsize=10, labelpad=8)
        ax.set_zlabel('Probability', fontsize=10, labelpad=8)
        ax.set_title(f'{class_labels[idx]}', fontsize=12, fontweight='bold')

        ax.set_zlim([0, 1])
        ax.view_init(elev=25, azim=45)

    fig.suptitle(f'3D Probability Surfaces (Elevation={elevation_fixed:.1f}m)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('3d_all_classes.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_3d_interactive_slices(model, utme_range=(450000, 550000),
                                    utmn_range=(4450000, 4550000),
                                    elevation_range=(200, 400),
                                    n_points=40):
    """
    Create 3D visualization with multiple elevation slices stacked.

    Parameters:
    -----------
    model : GaussianProcessClassifier
        Trained GP model
    utme_range : tuple
        (min, max) UTME values
    utmn_range : tuple
        (min, max) UTMN values
    elevation_range : tuple
        (min, max) elevation values
    n_points : int
        Grid resolution
    """

    # Create meshgrid for UTME and UTMN
    utme_vals = np.linspace(utme_range[0], utme_range[1], n_points)
    utmn_vals = np.linspace(utmn_range[0], utmn_range[1], n_points)
    UTME, UTMN = np.meshgrid(utme_vals, utmn_vals)

    # Create multiple elevation slices
    elevation_slices = np.linspace(elevation_range[0], elevation_range[1], 5)

    class_labels = ['Cedar Valley', 'Wapsipinicon', 'None']

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # For each elevation slice
    for elev_val in elevation_slices:
        # Flatten for prediction
        utme_flat = UTME.flatten()
        utmn_flat = UTMN.flatten()
        elevation_flat = np.full(len(utme_flat), elev_val)

        # Create feature matrix
        X_viz = np.column_stack([utme_flat, utmn_flat, elevation_flat])

        # Get predictions for first class
        probabilities = model.predict_proba(X_viz)
        prob_grid = probabilities[:, 0].reshape(UTME.shape)

        # Plot as contourf at this elevation
        norm = plt.Normalize(vmin=0, vmax=1)
        colors = cm.viridis(norm(prob_grid))

        ax.plot_surface(UTME, UTMN, np.full_like(UTME, elev_val),
                        facecolors=colors, alpha=0.6, shade=False)

    ax.set_xlabel('UTME (m)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('UTMN (m)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Elevation (m)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(f'3D Probability Slices: {class_labels[0]}',
                 fontsize=13, fontweight='bold', pad=20)

    # Add colorbar
    m = cm.ScalarMappable(cmap='viridis', norm=norm)
    m.set_array([])
    cbar = plt.colorbar(m, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Probability', fontsize=12, fontweight='bold')

    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('3d_elevation_slices.png', dpi=300, bbox_inches='tight')
    plt.show()
