// main.cpp - Research-grade usage example
#include "PlanarUnitDistance.hpp"
#include <iostream>
#include <iomanip>

int main() {
    using namespace PlanarUnitDistance;
    
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Planar Unit Distance Problem - Evolutionary Optimizer    ║\n";
    std::cout << "║                     Research-Grade Tool                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    // Test different configurations
    std::vector<int> pointCounts = {100, 500, 1000, 5000};
    
    for (int n : pointCounts) {
        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "Optimizing for " << n << " points\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
        
        // Configure optimizer
        OptimizationConfig cfg;
        cfg.populationSize = std::min(500, n * 2);
        cfg.eliteCount = cfg.populationSize / 10;
        cfg.maxGenerations = 2000;
        cfg.stagnationLimit = 300;
        cfg.mutationRate = 0.15;
        cfg.mutationStrength = 0.08;
        cfg.searchBoundX = std::sqrt(n) * 0.8;
        cfg.searchBoundY = std::sqrt(n) * 0.8;
        cfg.targetDistance = 1.0;
        cfg.minDistance = 0.05;
        
        // Run optimizer
        UnitDistanceOptimizer optimizer(cfg);
        PointSet best = optimizer.optimize(n);
        
        // Compare with baseline
        UnitDistanceSolver baseline;
        auto squareGrid = baseline.constructSquareGrid(static_cast<int>(std::sqrt(n)));
        double baselineDensity = baseline.computeDensity(squareGrid);
        
        double optimizedDensity = best.getFitness() / n;
        double improvement = (optimizedDensity - baselineDensity) / baselineDensity * 100.0;
        
        std::cout << "\n📊 Results Comparison:\n";
        std::cout << "   Square grid density:     " << std::fixed << std::setprecision(4) 
                  << baselineDensity << "\n";
        std::cout << "   Optimized density:       " << optimizedDensity << "\n";
        std::cout << "   Improvement:             " << improvement << "%\n";
        std::cout << "   Total unit distances:    " << best.getEdges().size() << "\n\n";
        
        // Export results
        std::string prefix = "results_n" + std::to_string(n);
        optimizer.exportResults(prefix + "_full.txt");
        exportToCSV(best, prefix + "_points.csv");
        exportEdgesToCSV(best, prefix + "_edges.csv");
        
        std::cout << "   Results saved to: " << prefix << "_*\n";
    }
    
    // Large-scale optimization (for Pi 5)
    std::cout << "\n\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Large-Scale Optimization (Raspberry Pi 5)         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    OptimizationConfig pi5cfg;
    pi5cfg.populationSize = 200;
    pi5cfg.eliteCount = 20;
    pi5cfg.maxGenerations = 1000;
    pi5cfg.parallelEvaluation = true;
    pi5cfg.threads = 4;
    pi5cfg.useNEON = true;
    
    UnitDistanceOptimizer pi5optimizer(pi5cfg);
    PointSet largeSet = pi5optimizer.optimize(10000);
    
    std::cout << "\n✅ Large-scale optimization complete!\n";
    std::cout << "   Final density: " << largeSet.getFitness() / 10000 << "\n";
    std::cout << "   This demonstrates edge-AI capability on Pi 5 hardware\n";
    
    return 0;
}