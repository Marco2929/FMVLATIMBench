# Usage Example

    # 1. Create dummy files for demonstration (You skip this if you have real images)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Create a "Goal" image (Checkpoint 1)
    img_goal = Image.new('RGB', (100, 100), color='red')
    img_goal.save("checkpoints/1.png")
    
    # Create a "Goal" image (Checkpoint 2)
    img_goal_2 = Image.new('RGB', (100, 100), color='blue')
    img_goal_2.save("checkpoints/2.png")

    # 2. Initialize the system
    scorer = VGBenchScorer("checkpoints")
    
    # 3. Simulate the Game Loop
    print("\n--- Starting Game Simulation ---")
    
    # Sim 1: Totally different image (Black screen)
    current_frame = Image.new('RGB', (100, 100), color='black')
    found, cp, dist = scorer.update(current_frame)
    print(f"Frame 1: Found? {found}")

    # Sim 2: Image that is 'red', but slightly different shade (Simulating artifact/noise)
    # The perceptual hash should still match this as 'red-ish' enough.
    current_frame = Image.new('RGB', (100, 100), color=(250, 10, 10)) 
    found, cp, dist = scorer.update(current_frame)
    print(f"Frame 2: Found? {found} (Checkpoint {cp}, Distance {dist})")
    
    # Sim 3: Now we see the blue image (Checkpoint 2)
    current_frame = Image.new('RGB', (100, 100), color='blue')
    found, cp, dist = scorer.update(current_frame)
    print(f"Frame 3: Found? {found} (Checkpoint {cp}, Distance {dist})")