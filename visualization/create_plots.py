import matplotlib.pyplot as plt

def fusion_method_comparison():
    methods = ['concat', 'add']
    force_losses = [0.559724, 0.593616]
    velocity_losses = [0.243279, 0.400446]

    # plot setup
    x = range(len(methods))
    width = 0.35

    plt.figure()
    plt.bar([i - width/2 for i in x], force_losses, width=width, label='Force')
    plt.bar([i + width/2 for i in x], velocity_losses, width=width, label='Velocity')
    plt.xticks(x, methods)
    plt.ylabel('Squared Error')
    plt.title('Comparison of squared error by fusion method')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fusion_method_comparison.png')

def modality_comparison():
    methods_force = ['both', 'force']
    methods_velocity = ['both', 'velocity']
    force_losses = [0.559724, 0.560472]
    velocity_losses = [0.243279, 0.211145]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    max_value = max(max(force_losses), max(velocity_losses))
    y_limit = max_value * 1.1
    
    ax1.bar(methods_force, force_losses, color='red', alpha=0.7)
    ax1.set_ylabel('Squared Error')
    ax1.set_title('Force Comparison')
    ax1.set_ylim(0, y_limit)

    ax2.bar(methods_velocity, velocity_losses, color='blue', alpha=0.7)
    ax2.set_ylabel('Squared Error')
    ax2.set_title('Velocity Comparison')
    ax2.set_ylim(0, y_limit)
    

    plt.suptitle('Joint training vs. modality-specific training', fontsize=14)
    plt.tight_layout()
    plt.savefig('modality_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def baseline_comparison():
    methods = ['diffusion', 'MLP baseline']
    force_losses = [0.559724, 0.299595]
    velocity_losses = [0.243279, 0.137249]
    
    x = range(len(methods))
    width = 0.35

    plt.figure()
    plt.bar([i - width/2 for i in x], force_losses, width=width, label='Force')
    plt.bar([i + width/2 for i in x], velocity_losses, width=width, label='Velocity')
    plt.xticks(x, methods)
    plt.ylabel('Squared Error')
    plt.title('Comparison of squared error by model type')
    plt.legend()
    plt.tight_layout()
    plt.savefig('baseline_comparison.png')
    plt.show()

def baseline_comparison_ik():
    methods = ['diffusion', 'MLP baseline']
    action_losses = [0.146685, 0.004158]
    
    x = range(len(methods))
    width = 0.35

    plt.figure()
    plt.bar([i - width/2 for i in x], action_losses, width=width, label='Action')
    plt.xticks(x, methods)
    plt.ylabel('Squared Error')
    plt.title('Comparison of action prediction error by model type')
    plt.ylim(0, 0.25)
    plt.tight_layout()
    plt.savefig('baseline_comparison_ik.png')
    plt.show()

if __name__ == "__main__":
    modality_comparison()
    baseline_comparison()
    baseline_comparison_ik()
    fusion_method_comparison()