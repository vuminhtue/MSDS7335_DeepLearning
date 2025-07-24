#!/usr/bin/env python3
"""
Main script to run MNIST to Letters Transfer Learning

This script provides options to run different transfer learning approaches:
1. Multi-class transfer learning (A, B, C, D, E classification)
2. Binary transfer learning (A vs NotA, B vs NotB, etc.)
3. Both approaches for comparison

Usage:
    python run_transfer_learning.py --approach multiclass
    python run_transfer_learning.py --approach binary  
    python run_transfer_learning.py --approach both
"""

import argparse
import os
import sys
import time

# Import our transfer learning modules
from mnist_to_letters_transfer import MNISTToLettersTransfer
from binary_transfer_learning import BinaryTransferLearning

def run_multiclass_approach():
    """Run the multi-class transfer learning approach."""
    print("🎯 RUNNING MULTI-CLASS TRANSFER LEARNING")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create and run multi-class pipeline
    transfer_pipeline = MNISTToLettersTransfer(data_dir="Images")
    model = transfer_pipeline.run_complete_pipeline()
    
    # Save model
    model.save('mnist_to_letters_multiclass_model.h5')
    
    end_time = time.time()
    print(f"\n⏱️  Multi-class training completed in {end_time - start_time:.2f} seconds")
    
    return transfer_pipeline

def run_binary_approach():
    """Run the binary transfer learning approach."""
    print("🎯 RUNNING BINARY TRANSFER LEARNING")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create and run binary pipeline
    binary_pipeline = BinaryTransferLearning(data_dir="Images")
    results = binary_pipeline.run_binary_pipeline()
    
    # Save models
    for letter in binary_pipeline.letters:
        if letter in binary_pipeline.binary_models:
            model_path = f'mnist_to_letters_binary_{letter}.h5'
            binary_pipeline.binary_models[letter].save(model_path)
    
    end_time = time.time()
    print(f"\n⏱️  Binary training completed in {end_time - start_time:.2f} seconds")
    
    return binary_pipeline

def compare_approaches():
    """Run both approaches and compare results."""
    print("🔄 RUNNING BOTH APPROACHES FOR COMPARISON")
    print("=" * 80)
    
    # Run both approaches
    print("\n📊 Phase 1: Multi-class Approach")
    multiclass_pipeline = run_multiclass_approach()
    
    print("\n📊 Phase 2: Binary Approach")
    binary_pipeline = run_binary_approach()
    
    # Compare predictions on test images
    print("\n" + "=" * 80)
    print("🆚 COMPARISON OF PREDICTIONS")
    print("=" * 80)
    
    test_images = [
        "Images/A/A1.jpg",
        "Images/B/B1.jpg", 
        "Images/C/C1.jpg",
        "Images/D/D6.jpg",
        "Images/E/E1.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📷 Image: {img_path}")
            try:
                # Multi-class prediction
                mc_letter, mc_probs, mc_confidence = multiclass_pipeline.predict_letter(img_path)
                print(f"   Multi-class: {mc_letter} (confidence: {mc_confidence:.4f})")
                
                # Binary prediction
                bin_letter, bin_scores = binary_pipeline.predict_letter_ensemble(img_path)
                max_bin_score = max(bin_scores.values())
                print(f"   Binary:      {bin_letter} (confidence: {max_bin_score:.4f})")
                
                # Agreement check
                agreement = "✅ AGREE" if mc_letter == bin_letter else "❌ DISAGREE"
                print(f"   Agreement:   {agreement}")
                
            except Exception as e:
                print(f"   Error: {e}")

def validate_environment():
    """Validate that the environment is set up correctly."""
    print("🔍 VALIDATING ENVIRONMENT")
    print("-" * 40)
    
    # Check if Images directory exists
    if not os.path.exists("Images"):
        print("❌ Error: 'Images' directory not found!")
        print("   Please ensure the Images folder with A, B, C, D, E subfolders exists.")
        return False
    
    # Check for letter directories
    letters = ['A', 'B', 'C', 'D', 'E']
    not_letters = ['NotA', 'NotB', 'NotC', 'NotD', 'NotE']
    
    missing_dirs = []
    for letter in letters + not_letters:
        letter_dir = os.path.join("Images", letter)
        if not os.path.exists(letter_dir):
            missing_dirs.append(letter)
    
    if missing_dirs:
        print(f"❌ Error: Missing directories: {missing_dirs}")
        return False
    
    # Check if directories have images
    total_images = 0
    for letter in letters:
        letter_dir = os.path.join("Images", letter)
        if os.path.exists(letter_dir):
            img_count = len([f for f in os.listdir(letter_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_images += img_count
            print(f"   {letter}: {img_count} images")
    
    if total_images == 0:
        print("❌ Error: No image files found in letter directories!")
        return False
    
    print(f"✅ Environment valid! Found {total_images} total images")
    return True

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="MNIST to Letters Transfer Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_transfer_learning.py --approach multiclass
  python run_transfer_learning.py --approach binary
  python run_transfer_learning.py --approach both
        """
    )
    
    parser.add_argument(
        '--approach', 
        choices=['multiclass', 'binary', 'both'],
        default='both',
        help='Transfer learning approach to use (default: both)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip environment validation'
    )
    
    args = parser.parse_args()
    
    print("🚀 MNIST TO LETTERS TRANSFER LEARNING")
    print("=" * 80)
    print(f"Approach: {args.approach}")
    print("=" * 80)
    
    # Validate environment
    if not args.skip_validation:
        if not validate_environment():
            print("\n❌ Environment validation failed. Exiting.")
            sys.exit(1)
        print()
    
    # Run selected approach
    try:
        if args.approach == 'multiclass':
            run_multiclass_approach()
            
        elif args.approach == 'binary':
            run_binary_approach()
            
        elif args.approach == 'both':
            compare_approaches()
        
        print("\n🎉 TRANSFER LEARNING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Show saved models
        print("\n📁 SAVED MODELS:")
        for filename in os.listdir('.'):
            if filename.endswith('.h5'):
                print(f"   - {filename}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 