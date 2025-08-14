import unittest
import tempfile
import os
import shutil
import json
import random
from typing import Any, Dict
from unittest.mock import patch, MagicMock
from openevolve.database.database import ProgramDatabase, Program


class TestProgram(unittest.TestCase):
    """Test the Program dataclass"""

    def test_program_creation(self):
        """Test creating a Program with minimal parameters"""
        program = Program(id="test_id", code="print('hello')")
        self.assertEqual(program.id, "test_id")
        self.assertEqual(program.code, "print('hello')")
        self.assertEqual(program.language, "python")
        self.assertEqual(program.generation, 0)
        self.assertIsInstance(program.timestamp, float)
        self.assertEqual(program.metrics, {})

    def test_program_with_metrics(self):
        """Test creating a Program with metrics"""
        metrics = {"score": 0.95, "complexity": 0.5}
        program = Program(id="test_id", code="print('hello')", metrics=metrics, generation=5)
        self.assertEqual(program.metrics, metrics)
        self.assertEqual(program.generation, 5)

    def test_program_to_dict(self):
        """Test converting Program to dictionary"""
        program = Program(id="test_id", code="print('hello')", metrics={"score": 0.95})
        program_dict = program.to_dict()
        self.assertIsInstance(program_dict, dict)
        self.assertEqual(program_dict["id"], "test_id")
        self.assertEqual(program_dict["code"], "print('hello')")
        self.assertEqual(program_dict["metrics"], {"score": 0.95})

    def test_program_from_dict(self):
        """Test creating Program from dictionary"""
        data = {
            "id": "test_id",
            "code": "print('hello')",
            "metrics": {"score": 0.95},
            "generation": 3,
            "unknown_field": "should_be_filtered",  # This should be filtered out
        }
        program = Program.from_dict(data)
        self.assertEqual(program.id, "test_id")
        self.assertEqual(program.code, "print('hello')")
        self.assertEqual(program.metrics, {"score": 0.95})
        self.assertEqual(program.generation, 3)
        # unknown_field should not be present
        self.assertFalse(hasattr(program, "unknown_field"))


class TestProgramDatabase(unittest.TestCase):
    """Test the ProgramDatabase class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_db")

    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_database_initialization(self):
        """Test database initialization with default parameters"""
        db = ProgramDatabase()
        self.assertIsNone(db.db_path)
        self.assertTrue(db.in_memory)
        self.assertEqual(db.population_size, 1000)
        self.assertEqual(db.archive_size, 100)
        self.assertEqual(db.num_islands, 5)
        self.assertEqual(len(db.islands), 5)
        self.assertEqual(db.feature_dimensions, ["complexity", "diversity"])
        self.assertEqual(len(db.programs), 0)

    def test_database_initialization_with_custom_params(self):
        """Test database initialization with custom parameters"""
        feature_dims = ["score", "complexity"]
        db = ProgramDatabase(
            db_path=self.db_path,
            population_size=500,
            archive_size=50,
            num_islands=3,
            feature_dimensions=feature_dims,
            random_seed=123,
        )
        self.assertEqual(db.db_path, self.db_path)
        self.assertEqual(db.population_size, 500)
        self.assertEqual(db.archive_size, 50)
        self.assertEqual(db.num_islands, 3)
        self.assertEqual(len(db.islands), 3)
        self.assertEqual(db.feature_dimensions, feature_dims)
        self.assertEqual(db.random_seed, 123)

    def test_add_program(self):
        """Test adding a program to the database"""
        db = ProgramDatabase(random_seed=42)
        program = Program(
            id="test_prog_1", code="print('hello')", metrics={"score": 0.8, "complexity": 0.5}
        )

        program_id = db.add(program, iteration=1)
        self.assertEqual(program_id, "test_prog_1")
        self.assertIn("test_prog_1", db.programs)
        self.assertEqual(db.programs["test_prog_1"], program)
        self.assertEqual(db.last_iteration, 1)
        self.assertEqual(program.iteration_found, 1)

    def test_get_program(self):
        """Test retrieving a program by ID"""
        db = ProgramDatabase()
        program = Program(id="test_prog_1", code="print('hello')")
        db.add(program)

        retrieved = db.get("test_prog_1")
        self.assertEqual(retrieved, program)

        # Test getting non-existent program
        non_existent = db.get("non_existent")
        self.assertIsNone(non_existent)

    def test_best_program_tracking(self):
        """Test best program tracking functionality"""
        db = ProgramDatabase(random_seed=42)

        # Add first program
        prog1 = Program(id="prog1", code="print('hello')", metrics={"combined_score": 0.5})
        db.add(prog1)
        self.assertEqual(db.get_best_program_id(), "prog1")

        # Add better program
        prog2 = Program(id="prog2", code="print('world')", metrics={"combined_score": 0.8})
        db.add(prog2)
        self.assertEqual(db.get_best_program_id(), "prog2")

        # Add worse program - best should not change
        prog3 = Program(id="prog3", code="print('test')", metrics={"combined_score": 0.3})
        db.add(prog3)
        self.assertEqual(db.get_best_program_id(), "prog2")

    def test_get_best_program(self):
        """Test getting the best program"""
        db = ProgramDatabase(random_seed=42)

        # Empty database
        self.assertIsNone(db.get_best_program())

        # Add programs with different scores
        prog1 = Program(id="prog1", code="code1", metrics={"combined_score": 0.5})
        prog2 = Program(id="prog2", code="code2", metrics={"combined_score": 0.8})
        prog3 = Program(id="prog3", code="code3", metrics={"combined_score": 0.3})

        db.add(prog1)
        db.add(prog2)
        db.add(prog3)

        best = db.get_best_program()
        self.assertIsNotNone(best)
        if best is not None:  # Type guard
            self.assertEqual(best.id, "prog2")
            self.assertEqual(best.metrics["combined_score"], 0.8)

    def test_get_top_programs(self):
        """Test getting top N programs"""
        db = ProgramDatabase(random_seed=42)

        # Add programs with different scores
        programs = []
        for i in range(5):
            prog = Program(id=f"prog{i}", code=f"code{i}", metrics={"combined_score": i * 0.2})
            programs.append(prog)
            db.add(prog)

        # Get top 3 programs
        top_3 = db.get_top_programs(n=3)
        self.assertEqual(len(top_3), 3)
        # Should be sorted by score descending
        self.assertEqual(top_3[0].id, "prog4")  # highest score (0.8)
        self.assertEqual(top_3[1].id, "prog3")  # second highest (0.6)
        self.assertEqual(top_3[2].id, "prog2")  # third highest (0.4)

    def test_island_management(self):
        """Test island-based program management"""
        db = ProgramDatabase(num_islands=3, random_seed=42)

        # Test initial state
        self.assertEqual(db.get_current_island(), 0)
        self.assertEqual(len(db.islands), 3)

        # Add program to current island
        prog = Program(id="prog1", code="code1", metrics={"score": 0.5})
        db.add(prog)
        self.assertIn("prog1", db.islands[0])
        self.assertEqual(prog.metadata["island"], 0)

        # Switch to next island
        next_island = db.next_island()
        self.assertEqual(next_island, 1)
        self.assertEqual(db.get_current_island(), 1)

        # Add program to new island
        prog2 = Program(id="prog2", code="code2", metrics={"score": 0.7})
        db.add(prog2)
        self.assertIn("prog2", db.islands[1])
        self.assertEqual(prog2.metadata["island"], 1)

    def test_population_limit_enforcement(self):
        """Test that population size limits are enforced"""
        db = ProgramDatabase(population_size=3, random_seed=42)

        # Add programs up to limit
        for i in range(5):
            prog = Program(id=f"prog{i}", code=f"code{i}", metrics={"combined_score": i * 0.1})
            db.add(prog)

        # Should only keep 3 programs (the best ones)
        self.assertEqual(len(db.programs), 3)

        # Best programs should be kept
        remaining_ids = set(db.programs.keys())
        # Should keep prog4 (0.4), prog3 (0.3), prog2 (0.2) - the highest scoring ones
        expected_remaining = {"prog4", "prog3", "prog2"}
        self.assertEqual(remaining_ids, expected_remaining)

    def test_sampling(self):
        """Test program sampling functionality"""
        db = ProgramDatabase(num_inspirations=2, random_seed=42)

        # Add some programs
        for i in range(5):
            prog = Program(id=f"prog{i}", code=f"code{i}", metrics={"combined_score": i * 0.2})
            db.add(prog)

        # Test sampling
        parent, inspirations = db.sample()
        self.assertIsInstance(parent, Program)
        self.assertIsInstance(inspirations, list)
        self.assertTrue(len(inspirations) <= 2)  # num_inspirations

        # Parent should be a valid program
        self.assertIn(parent.id, db.programs)

    def test_save_and_load(self):
        """Test saving and loading database"""
        db = ProgramDatabase(db_path=self.db_path, random_seed=42)

        # Add some programs
        prog1 = Program(id="prog1", code="code1", metrics={"score": 0.5})
        prog2 = Program(id="prog2", code="code2", metrics={"score": 0.8})
        db.add(prog1)
        db.add(prog2)

        # Save database
        db.save(iteration=10)

        # Create new database and load
        db2 = ProgramDatabase(db_path=self.db_path)

        # Check that programs were loaded
        self.assertEqual(len(db2.programs), 2)
        self.assertIn("prog1", db2.programs)
        self.assertIn("prog2", db2.programs)
        self.assertEqual(db2.last_iteration, 10)
        self.assertEqual(db2.programs["prog1"].code, "code1")
        self.assertEqual(db2.programs["prog2"].metrics["score"], 0.8)

    def test_feature_map_functionality(self):
        """Test MAP-Elites feature mapping"""
        db = ProgramDatabase(
            feature_dimensions=["complexity", "diversity"], feature_bins=5, random_seed=42
        )

        # Add a program
        prog = Program(id="prog1", code="print('hello world')", metrics={"score": 0.5})
        db.add(prog)

        # Should have created feature mapping
        self.assertTrue(len(db.feature_map) > 0)

        # Feature map should contain our program
        feature_keys = list(db.feature_map.keys())
        self.assertEqual(db.feature_map[feature_keys[0]], "prog1")

    def test_archive_management(self):
        """Test archive functionality"""
        db = ProgramDatabase(archive_size=2, random_seed=42)

        # Add programs
        for i in range(4):
            prog = Program(id=f"prog{i}", code=f"code{i}", metrics={"combined_score": i * 0.25})
            db.add(prog)

        # Archive should contain best programs and be limited to archive_size
        self.assertTrue(len(db.archive) <= 2)

    def test_island_stats(self):
        """Test getting island statistics"""
        db = ProgramDatabase(num_islands=2, random_seed=42)

        # Add programs to different islands
        prog1 = Program(id="prog1", code="code1", metrics={"combined_score": 0.5})
        db.add(prog1, target_island=0)

        db.set_current_island(1)
        prog2 = Program(id="prog2", code="code2", metrics={"combined_score": 0.8})
        db.add(prog2, target_island=1)

        stats = db.get_island_stats()
        self.assertEqual(len(stats), 2)
        self.assertIsInstance(stats[0], dict)
        self.assertIn("island", stats[0])  # The actual field name is "island", not "island_id"
        self.assertIn("population_size", stats[0])

    def test_migration_conditions(self):
        """Test migration condition checking"""
        db = ProgramDatabase(migration_interval=5, num_islands=2)

        # Initially should not migrate
        self.assertFalse(db.should_migrate())

        # Increment generations
        for i in range(6):
            db.increment_island_generation(0)

        # Should now trigger migration
        self.assertTrue(db.should_migrate())

    def test_get_top_programs_by_island(self):
        """Test getting top programs from specific island"""
        db = ProgramDatabase(num_islands=2, random_seed=42)

        # Add programs to island 0
        prog1 = Program(id="prog1", code="code1", metrics={"combined_score": 0.5})
        prog2 = Program(id="prog2", code="code2", metrics={"combined_score": 0.8})
        db.add(prog1, target_island=0)
        db.add(prog2, target_island=0)

        # Add program to island 1
        prog3 = Program(id="prog3", code="code3", metrics={"combined_score": 0.9})
        db.add(prog3, target_island=1)

        # Get top programs from island 0 only
        top_island_0 = db.get_top_programs(n=5, island_idx=0)
        self.assertEqual(len(top_island_0), 2)  # Only 2 programs in island 0
        self.assertEqual(top_island_0[0].id, "prog2")  # Higher score
        self.assertEqual(top_island_0[1].id, "prog1")

        # Get top programs from island 1 only
        top_island_1 = db.get_top_programs(n=5, island_idx=1)
        self.assertEqual(len(top_island_1), 1)
        self.assertEqual(top_island_1[0].id, "prog3")

    def test_invalid_island_index(self):
        """Test handling of invalid island indices"""
        db = ProgramDatabase(num_islands=2)

        # Should raise IndexError for invalid island index
        with self.assertRaises(IndexError):
            db.get_top_programs(n=5, island_idx=5)

        with self.assertRaises(IndexError):
            db.get_top_programs(n=5, island_idx=-1)

    def test_program_replacement_in_feature_map(self):
        """Test that better programs replace worse ones in the same feature cell"""
        db = ProgramDatabase(feature_dimensions=["complexity"], feature_bins=2, random_seed=42)

        # Add program with low score
        prog1 = Program(
            id="prog1",
            code="x=1",  # Short code for same complexity bin
            metrics={"combined_score": 0.3},
        )
        db.add(prog1)

        # Add program with similar complexity but higher score
        prog2 = Program(
            id="prog2",
            code="y=2",  # Similar length, same complexity bin
            metrics={"combined_score": 0.8},
        )
        db.add(prog2)

        # Feature map should contain the better program
        # Since both programs likely map to the same feature cell,
        # the feature map should contain prog2 (better score)
        feature_values = list(db.feature_map.values())
        if len(feature_values) == 1:
            # Programs mapped to same cell, better one should remain
            self.assertEqual(feature_values[0], "prog2")


class TestProgramDatabaseAdvanced(unittest.TestCase):
    """Advanced test cases for ProgramDatabase functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_advanced_db")

    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_database_initialization_with_all_parameters(self):
        """Test database initialization with comprehensive parameter set"""
        db = ProgramDatabase(
            db_path=self.db_path,
            in_memory=False,
            log_prompts=False,
            population_size=200,
            archive_size=20,
            num_islands=4,
            elite_selection_ratio=0.15,
            exploration_ratio=0.3,
            exploitation_ratio=0.6,
            diversity_metric="jaccard",
            feature_dimensions=["complexity", "diversity", "performance"],
            feature_bins=8,
            diversity_reference_size=15,
            migration_interval=25,
            migration_rate=0.2,
            random_seed=999,
            artifacts_base_path=self.temp_dir,
            artifact_size_threshold=16 * 1024,
            cleanup_old_artifacts=False,
            artifact_retention_days=7,
            num_inspirations=3,
        )

        # Verify all parameters were set correctly
        self.assertEqual(db.db_path, self.db_path)
        self.assertFalse(db.in_memory)
        self.assertFalse(db.log_prompts)
        self.assertEqual(db.population_size, 200)
        self.assertEqual(db.archive_size, 20)
        self.assertEqual(db.num_islands, 4)
        self.assertEqual(db.elite_selection_ratio, 0.15)
        self.assertEqual(db.exploration_ratio, 0.3)
        self.assertEqual(db.exploitation_ratio, 0.6)
        self.assertEqual(db.diversity_metric, "jaccard")
        self.assertEqual(db.feature_dimensions, ["complexity", "diversity", "performance"])
        self.assertEqual(db.feature_bins, 8)
        self.assertEqual(db.diversity_reference_size, 15)
        self.assertEqual(db.migration_interval, 25)
        self.assertEqual(db.migration_rate, 0.2)
        self.assertEqual(db.random_seed, 999)
        self.assertEqual(db.artifacts_base_path, self.temp_dir)
        self.assertEqual(db.artifact_size_threshold, 16 * 1024)
        self.assertFalse(db.cleanup_old_artifacts)
        self.assertEqual(db.artifact_retention_days, 7)
        self.assertEqual(db.num_inspirations, 3)

    def test_large_scale_program_addition(self):
        """Test adding many programs and verifying population limits"""
        db = ProgramDatabase(population_size=50, archive_size=10, random_seed=42)

        # Add 100 programs with varying scores
        added_programs = []
        for i in range(100):
            score = random.uniform(0.0, 1.0)
            prog = Program(
                id=f"large_prog_{i}",
                code=f"def func_{i}(): return {i}",
                metrics={"combined_score": score, "accuracy": score * 0.9},
            )
            added_programs.append((prog.id, score))
            db.add(prog, iteration=i)

        # Should maintain population limit
        self.assertEqual(len(db.programs), 50)

        # Archive should contain best programs
        self.assertLessEqual(len(db.archive), 10)

        # Best program should be tracked correctly
        best_program = db.get_best_program()
        self.assertIsNotNone(best_program)

        # Verify the best program has highest score among remaining
        all_scores = [p.metrics.get("combined_score", 0) for p in db.programs.values()]
        if best_program is not None:  # Type guard
            self.assertEqual(best_program.metrics["combined_score"], max(all_scores))

    def test_island_isolation_and_migration(self):
        """Test that islands maintain separate populations and migrate correctly"""
        db = ProgramDatabase(
            num_islands=3,
            population_size=30,  # 10 per island effectively
            migration_interval=5,
            migration_rate=0.3,
            random_seed=42,
        )

        # Add programs to each island separately
        island_programs = {0: [], 1: [], 2: []}

        for island in range(3):
            db.set_current_island(island)
            for i in range(10):
                prog = Program(
                    id=f"island_{island}_prog_{i}",
                    code=f"def island_{island}_func_{i}(): return {i}",
                    metrics={"combined_score": random.uniform(0.0, 1.0)},
                )
                island_programs[island].append(prog.id)
                db.add(prog, target_island=island)

        # Verify programs are on correct islands
        for island in range(3):
            island_set = db.islands[island]
            for prog_id in island_programs[island]:
                self.assertIn(prog_id, island_set)

        # Test migration trigger
        for island in range(3):
            for _ in range(6):  # Exceed migration_interval
                db.increment_island_generation(island)

        self.assertTrue(db.should_migrate())

    def test_feature_map_behavior_with_different_dimensions(self):
        """Test feature map with various feature dimensions"""
        # Test with single dimension
        db1 = ProgramDatabase(feature_dimensions=["complexity"], feature_bins=5, random_seed=42)

        prog1 = Program(
            id="single_dim_prog", code="print('simple')", metrics={"combined_score": 0.5}
        )
        db1.add(prog1)
        self.assertGreater(len(db1.feature_map), 0)

        # Test with multiple dimensions
        db2 = ProgramDatabase(
            feature_dimensions=["complexity", "diversity"],  # Use only supported dimensions
            feature_bins=3,
            random_seed=42,
        )

        prog2 = Program(
            id="multi_dim_prog",
            code="def complex_function(): return sum(range(100))",
            metrics={"combined_score": 0.8},
        )
        db2.add(prog2)
        self.assertGreater(len(db2.feature_map), 0)

    def test_database_persistence_comprehensive(self):
        """Test comprehensive save/load functionality with complex state"""
        db = ProgramDatabase(
            db_path=self.db_path, num_islands=3, population_size=15, archive_size=5, random_seed=123
        )

        # Create complex state
        programs_added = []
        for island in range(3):
            db.set_current_island(island)
            for i in range(5):
                prog = Program(
                    id=f"persist_island_{island}_prog_{i}",
                    code=f"def func(): return {island * 10 + i}",
                    metrics={
                        "combined_score": random.uniform(0.0, 1.0),
                        "complexity": random.uniform(0.0, 0.5),
                        "diversity": random.uniform(0.0, 0.8),
                    },
                    generation=i,
                    parent_id=f"parent_{i}" if i > 0 else None,
                )
                programs_added.append(prog)
                db.add(prog, iteration=island * 5 + i, target_island=island)

        # Set some island generations
        for island in range(3):
            for _ in range(island + 1):
                db.increment_island_generation(island)

        original_state = {
            "num_programs": len(db.programs),
            "num_archive": len(db.archive),
            "best_program_id": db.best_program_id,
            "current_island": db.current_island,
            "island_generations": db.island_generations.copy(),
            "last_iteration": db.last_iteration,
            "island_sizes": [len(island) for island in db.islands],
        }

        # Save database
        db.save(iteration=20)

        # Create new database and load
        db_loaded = ProgramDatabase(
            db_path=self.db_path, num_islands=3, random_seed=123  # Match original configuration
        )

        # Verify state restoration
        self.assertEqual(len(db_loaded.programs), original_state["num_programs"])
        self.assertEqual(len(db_loaded.archive), original_state["num_archive"])
        self.assertEqual(db_loaded.best_program_id, original_state["best_program_id"])
        # Note: island_generations might be reset on load, which is okay for persistence
        self.assertEqual(db_loaded.last_iteration, 20)  # Updated iteration

        # Verify all programs loaded correctly
        for prog in programs_added:
            loaded_prog = db_loaded.get(prog.id)
            self.assertIsNotNone(loaded_prog, f"Program {prog.id} should be loaded")
            if loaded_prog is not None:  # Type guard
                self.assertEqual(loaded_prog.code, prog.code)
                self.assertEqual(loaded_prog.metrics, prog.metrics)

    def test_sampling_strategies_comprehensive(self):
        """Test different sampling strategies and edge cases"""
        db = ProgramDatabase(
            num_islands=2,
            population_size=20,
            num_inspirations=3,
            elite_selection_ratio=0.2,
            exploration_ratio=0.3,
            exploitation_ratio=0.5,
            random_seed=42,
        )

        # Add diverse programs
        high_score_progs = []
        med_score_progs = []
        low_score_progs = []

        for i in range(20):
            if i < 5:
                score = random.uniform(0.8, 1.0)
                prog_list = high_score_progs
            elif i < 15:
                score = random.uniform(0.4, 0.7)
                prog_list = med_score_progs
            else:
                score = random.uniform(0.0, 0.3)
                prog_list = low_score_progs

            prog = Program(
                id=f"sampling_prog_{i}",
                code=f"def func_{i}(): return {i}",
                metrics={"combined_score": score},
            )
            prog_list.append(prog.id)
            db.add(prog, target_island=i % 2)

        # Test multiple sampling rounds
        sample_results = []
        for _ in range(50):  # Multiple samples to test distribution
            parent, inspirations = db.sample()
            sample_results.append(
                {
                    "parent_id": parent.id,
                    "parent_score": parent.metrics.get("combined_score", 0),
                    "num_inspirations": len(inspirations),
                    "inspiration_scores": [
                        insp.metrics.get("combined_score", 0) for insp in inspirations
                    ],
                }
            )

        # Verify sampling properties
        self.assertEqual(len(sample_results), 50)

        # All samples should have a parent
        self.assertTrue(all(result["parent_id"] for result in sample_results))

        # Inspiration counts should be reasonable
        inspiration_counts = [result["num_inspirations"] for result in sample_results]
        self.assertTrue(all(count >= 0 for count in inspiration_counts))
        self.assertTrue(all(count <= 3 for count in inspiration_counts))  # num_inspirations limit

    def test_metrics_aggregation_and_scoring(self):
        """Test how database handles different metric types and aggregation"""
        db = ProgramDatabase(random_seed=42)

        # Add programs with different metric structures
        # Design scores so that the average will rank them correctly
        prog1 = Program(
            id="metrics_prog_1",
            code="simple_code",
            metrics={
                "combined_score": 0.7,
                "accuracy": 0.85,
                "precision": 0.75,
                "recall": 0.80,
                "f1_score": 0.77,
            },
        )

        prog2 = Program(
            id="metrics_prog_2",
            code="complex_code",
            metrics={
                "combined_score": 0.6,
                "accuracy": 0.50,  # Lower values to make average lower
                "execution_time": 0.5,
                "memory_usage": 0.4,
            },
        )

        prog3 = Program(
            id="metrics_prog_3",
            code="minimal_code",
            metrics={"combined_score": 0.9},  # Single high metric
        )

        db.add(prog1)
        db.add(prog2)
        db.add(prog3)

        # Verify best program selection works with different metric structures
        best = db.get_best_program()
        self.assertIsNotNone(best)
        if best is not None:  # Type guard for static analysis
            self.assertEqual(best.id, "metrics_prog_3")  # Highest combined_score

        # Verify top programs ranking by average score
        top_progs = db.get_top_programs(n=3)

        # Manually calculate expected averages to verify ranking
        prog1_avg = (0.7 + 0.85 + 0.75 + 0.80 + 0.77) / 5  # = 0.774
        prog2_avg = (0.6 + 0.50 + 0.5 + 0.4) / 4  # = 0.5
        prog3_avg = 0.9  # Single metric

        # Expected order: prog3 (0.9), prog1 (0.774), prog2 (0.5)
        expected_order = ["metrics_prog_3", "metrics_prog_1", "metrics_prog_2"]
        actual_order = [prog.id for prog in top_progs]
        self.assertEqual(actual_order, expected_order)

    def test_diversity_and_complexity_calculation(self):
        """Test diversity and complexity feature calculations"""
        db = ProgramDatabase(
            feature_dimensions=["complexity", "diversity"],
            diversity_reference_size=5,
            random_seed=42,
        )

        # Add programs with varying code complexity
        simple_prog = Program(id="simple", code="x = 1", metrics={"combined_score": 0.5})

        moderate_prog = Program(
            id="moderate",
            code="""
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
""",
            metrics={"combined_score": 0.6},
        )

        complex_prog = Program(
            id="complex",
            code="""
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    def process(self, data):
        if data in self.cache:
            return self.cache[data]
        
        result = []
        for item in data:
            if self.config.get('transform'):
                item = self.transform(item)
            if self.validate(item):
                result.append(item)
        
        self.cache[data] = result
        return result
    
    def transform(self, item):
        return item * 2
    
    def validate(self, item):
        return item > 0
""",
            metrics={"combined_score": 0.7},
        )

        db.add(simple_prog)
        db.add(moderate_prog)
        db.add(complex_prog)

        # Verify programs have different complexity values
        progs = [db.get("simple"), db.get("moderate"), db.get("complex")]

        # Filter out None values and check they all exist
        valid_progs = [prog for prog in progs if prog is not None]
        self.assertEqual(len(valid_progs), 3, "All programs should be retrievable")

        complexities = [prog.complexity for prog in valid_progs]

        # Should have increasing complexity (though exact values depend on implementation)
        self.assertTrue(complexities[0] <= complexities[1] <= complexities[2])

        # Verify feature map contains entries
        self.assertGreater(len(db.feature_map), 0)

    def test_concurrent_operations_simulation(self):
        """Test database behavior under simulated concurrent operations"""
        db = ProgramDatabase(population_size=100, archive_size=20, num_islands=4, random_seed=42)

        # Simulate concurrent additions from different "threads"
        all_programs = []
        for thread_id in range(4):
            for i in range(25):
                prog = Program(
                    id=f"concurrent_thread_{thread_id}_prog_{i}",
                    code=f"def thread_{thread_id}_func_{i}(): return {thread_id * 100 + i}",
                    metrics={"combined_score": random.uniform(0.0, 1.0)},
                    generation=i,
                    metadata={"thread_id": thread_id},
                )
                all_programs.append(prog)

                # Simulate different islands for different threads
                target_island = thread_id % db.num_islands
                db.add(prog, iteration=thread_id * 25 + i, target_island=target_island)

        # Verify final state consistency
        self.assertEqual(len(db.programs), 100)  # Population limit maintained

        # Verify island distribution
        total_island_programs = sum(len(island) for island in db.islands)
        self.assertEqual(total_island_programs, 100)

        # Verify best program tracking
        best_prog = db.get_best_program()
        self.assertIsNotNone(best_prog)

        # Verify all islands have some programs
        for i, island in enumerate(db.islands):
            self.assertGreater(len(island), 0, f"Island {i} should have programs")

    def test_edge_cases_and_error_handling(self):
        """Test database behavior with edge cases and potential error conditions"""
        db = ProgramDatabase(population_size=5, archive_size=2, num_islands=2, random_seed=42)

        # Test empty database operations
        self.assertIsNone(db.get_best_program())
        self.assertEqual(len(db.get_top_programs(n=10)), 0)

        # Test invalid program retrieval
        self.assertIsNone(db.get("nonexistent_id"))

        # Test sampling with empty database - should handle gracefully
        try:
            parent, inspirations = db.sample()
            # If it returns something, verify it's valid
            if parent is not None:
                self.assertIsInstance(parent, Program)
                self.assertIsInstance(inspirations, list)
        except Exception as e:
            # If it raises an exception, that's also acceptable for empty database
            # Include StopIteration as acceptable for empty database sampling
            self.assertIsInstance(e, (ValueError, IndexError, AttributeError, StopIteration))

        # Add one program and test single-program sampling
        single_prog = Program(
            id="single_prog", code="print('only one')", metrics={"combined_score": 0.5}
        )
        db.add(single_prog)

        # Now sampling should work
        parent, inspirations = db.sample()
        self.assertEqual(parent.id, "single_prog")
        self.assertIsInstance(inspirations, list)

        # Test invalid island operations
        with self.assertRaises(IndexError):
            db.get_top_programs(n=5, island_idx=999)


if __name__ == "__main__":
    unittest.main()
