// PlanarUnitDistance.hpp
#ifndef PLANAR_UNIT_DISTANCE_HPP
#define PLANAR_UNIT_DISTANCE_HPP

#include <vector>
#include <set>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>
#include <memory>
#include <chrono>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace PlanarUnitDistance {

// ============================================================================
// Core Data Structures
// ============================================================================

struct Point2D {
    double x, y;
    
    Point2D(double x_ = 0, double y_ = 0) : x(x_), y(y_) {}
    
    bool operator==(const Point2D& other) const {
        return std::abs(x - other.x) < 1e-9 && std::abs(y - other.y) < 1e-9;
    }
    
    bool operator<(const Point2D& other) const {
        if (std::abs(x - other.x) > 1e-9) return x < other.x;
        return y < other.y;
    }
    
    double distanceTo(const Point2D& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
    
    double distanceSqTo(const Point2D& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return dx * dx + dy * dy;
    }
    
    Point2D operator+(const Point2D& other) const { return Point2D(x + other.x, y + other.y); }
    Point2D operator-(const Point2D& other) const { return Point2D(x - other.x, y - other.y); }
    Point2D operator*(double s) const { return Point2D(x * s, y * s); }
};

struct Point2DHash {
    std::size_t operator()(const Point2D& p) const {
        auto h1 = std::hash<double>{}(std::round(p.x * 1e9));
        auto h2 = std::hash<double>{}(std::round(p.y * 1e9));
        return h1 ^ (h2 << 1);
    }
};

struct UnitDistanceGraph {
    std::vector<Point2D> points;
    std::vector<std::pair<int, int>> edges;
    
    int unitDistanceCount() const { return edges.size(); }
    double density() const { return points.empty() ? 0.0 : static_cast<double>(edges.size()) / points.size(); }
    
    void clear() {
        points.clear();
        edges.clear();
    }
};

// ============================================================================
// Optimization Parameters
// ============================================================================

struct OptimizationConfig {
    // Population parameters
    size_t populationSize = 200;
    size_t eliteCount = 20;
    size_t tournamentSize = 5;
    
    // Mutation parameters
    double mutationRate = 0.15;
    double mutationStrength = 0.08;
    double largeMutationRate = 0.05;
    double largeMutationStrength = 0.25;
    
    // Crossover parameters
    double crossoverRate = 0.7;
    
    // Geometric constraints
    double minDistance = 0.01;      // Minimum separation between points
    double targetDistance = 1.0;    // Target unit distance
    double distanceTolerance = 1e-6; // Tolerance for unit distance check
    
    // Annealing parameters
    double initialTemperature = 10.0;
    double coolingRate = 0.995;
    int iterationsPerTemp = 100;
    
    // Search bounds
    double searchBoundX = 10.0;
    double searchBoundY = 10.0;
    
    // Performance
    bool useNEON = true;
    bool parallelEvaluation = true;
    int threads = 4;
    
    // Termination
    int maxGenerations = 5000;
    int stagnationLimit = 200;
    double targetFitness = 0.0; // 0 = no target
};

// ============================================================================
// Spatial Acceleration Structure
// ============================================================================

class SpatialHashGrid {
private:
    double cellSize;
    std::unordered_map<std::pair<int, int>, std::vector<int>, 
        std::function<size_t(const std::pair<int,int>&)>> grid;
    
    size_t hashPair(const std::pair<int, int>& p) const {
        return std::hash<long long>()((static_cast<long long>(p.first) << 32) ^ p.second);
    }
    
public:
    SpatialHashGrid(double cellSz = 1.0) : cellSize(cellSz), 
        grid(1000, std::function<size_t(const std::pair<int,int>&)>(
            [this](const std::pair<int,int>& p) { return this->hashPair(p); })) {}
    
    std::pair<int, int> getCell(const Point2D& p) const {
        return {static_cast<int>(std::floor(p.x / cellSize)),
                static_cast<int>(std::floor(p.y / cellSize))};
    }
    
    void build(const std::vector<Point2D>& points) {
        grid.clear();
        for (size_t i = 0; i < points.size(); ++i) {
            auto cell = getCell(points[i]);
            grid[cell].push_back(i);
        }
    }
    
    std::vector<int> getNeighbors(const Point2D& p, int radius = 1) const {
        std::vector<int> neighbors;
        auto center = getCell(p);
        
        for (int dx = -radius; dx <= radius; ++dx) {
            for (int dy = -radius; dy <= radius; ++dy) {
                auto cell = std::make_pair(center.first + dx, center.second + dy);
                auto it = grid.find(cell);
                if (it != grid.end()) {
                    neighbors.insert(neighbors.end(), it->second.begin(), it->second.end());
                }
            }
        }
        
        return neighbors;
    }
};

// ============================================================================
// Individual (Point Set) for Evolutionary Optimization
// ============================================================================

class PointSet {
private:
    std::vector<Point2D> points;
    mutable std::vector<std::pair<int, int>> edges;
    mutable double fitness;
    mutable bool fitnessValid;
    mutable SpatialHashGrid spatialGrid;
    
    void computeEdges() const {
        if (!fitnessValid) return;
        
        edges.clear();
        spatialGrid.build(points);
        
        double targetSq = targetDistance * targetDistance;
        double tolSq = distanceTolerance * distanceTolerance;
        double minSq = minDistance * minDistance;
        
        #pragma omp parallel for if(points.size() > 1000)
        for (size_t i = 0; i < points.size(); ++i) {
            auto neighbors = spatialGrid.getNeighbors(points[i], 2);
            for (int idx : neighbors) {
                if (idx <= static_cast<int>(i)) continue;
                
                double dx = points[i].x - points[idx].x;
                double dy = points[i].y - points[idx].y;
                double d2 = dx*dx + dy*dy;
                
                // Check minimum separation constraint
                if (d2 < minSq) {
                    #pragma omp critical
                    fitness = -1e9; // Severe penalty
                    return;
                }
                
                // Check unit distance
                if (std::abs(d2 - targetSq) <= tolSq) {
                    #pragma omp critical
                    edges.emplace_back(i, idx);
                }
            }
        }
        
        fitnessValid = true;
    }
    
public:
    double targetDistance = 1.0;
    double distanceTolerance = 1e-6;
    double minDistance = 0.01;
    
    PointSet() : fitness(0), fitnessValid(false), spatialGrid(1.0) {}
    
    PointSet(const std::vector<Point2D>& pts) 
        : points(pts), fitness(0), fitnessValid(false), spatialGrid(1.0) {}
    
    // Copy constructor
    PointSet(const PointSet& other) 
        : points(other.points), fitness(other.fitness), 
          fitnessValid(other.fitnessValid), spatialGrid(1.0) {
        if (fitnessValid) {
            edges = other.edges;
        }
    }
    
    // Accessors
    const std::vector<Point2D>& getPoints() const { return points; }
    std::vector<Point2D>& getPoints() { fitnessValid = false; return points; }
    
    const std::vector<std::pair<int, int>>& getEdges() const {
        if (!fitnessValid) computeEdges();
        return edges;
    }
    
    double getFitness() const {
        if (!fitnessValid) computeEdges();
        return fitness;
    }
    
    size_t size() const { return points.size(); }
    
    void setPoint(size_t i, const Point2D& p) {
        points[i] = p;
        fitnessValid = false;
    }
    
    void addPoint(const Point2D& p) {
        points.push_back(p);
        fitnessValid = false;
    }
    
    void clear() {
        points.clear();
        edges.clear();
        fitness = 0;
        fitnessValid = false;
    }
    
    void evaluate() {
        fitnessValid = false;
        computeEdges();
        fitness = static_cast<double>(edges.size());
        
        // Additional geometric penalties
        double avgPenalty = 0.0;
        int invalidCount = 0;
        
        for (size_t i = 0; i < points.size(); ++i) {
            if (points[i].x < -searchBoundX || points[i].x > searchBoundX ||
                points[i].y < -searchBoundY || points[i].y > searchBoundY) {
                invalidCount++;
            }
        }
        
        if (invalidCount > 0) {
            fitness -= invalidCount * 10.0;
        }
    }
    
    // Operators for sorting
    bool operator<(const PointSet& other) const {
        return getFitness() < other.getFitness();
    }
    
    bool operator>(const PointSet& other) const {
        return getFitness() > other.getFitness();
    }
    
    // Parameters (need to be accessible for algorithms)
    double searchBoundX = 10.0;
    double searchBoundY = 10.0;
};

// ============================================================================
// Evolutionary Optimizer
// ============================================================================

class UnitDistanceOptimizer {
private:
    OptimizationConfig config;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniformDist;
    std::normal_distribution<double> normalDist;
    
    // Population
    std::vector<PointSet> population;
    PointSet bestSolution;
    double bestFitness;
    int generationsWithoutImprovement;
    
    // Statistics
    std::vector<double> fitnessHistory;
    std::vector<double> diversityHistory;
    
    // Helper functions
    double randomDouble(double min = 0.0, double max = 1.0) {
        return min + uniformDist(rng) * (max - min);
    }
    
    Point2D randomPoint() {
        return Point2D(
            randomDouble(-config.searchBoundX, config.searchBoundX),
            randomDouble(-config.searchBoundY, config.searchBoundY)
        );
    }
    
    void initializePopulation(int numPoints) {
        population.clear();
        population.reserve(config.populationSize);
        
        for (size_t i = 0; i < config.populationSize; ++i) {
            PointSet individual;
            individual.targetDistance = config.targetDistance;
            individual.distanceTolerance = config.distanceTolerance;
            individual.minDistance = config.minDistance;
            individual.searchBoundX = config.searchBoundX;
            individual.searchBoundY = config.searchBoundY;
            
            // Initialize with structured patterns + noise
            if (i < config.populationSize / 3) {
                // Square grid initialization
                int gridSize = static_cast<int>(std::sqrt(numPoints));
                for (int x = 0; x < gridSize && individual.size() < static_cast<size_t>(numPoints); ++x) {
                    for (int y = 0; y < gridSize && individual.size() < static_cast<size_t>(numPoints); ++y) {
                        double px = (x - gridSize/2.0) * 0.8;
                        double py = (y - gridSize/2.0) * 0.8;
                        px += randomDouble(-0.1, 0.1);
                        py += randomDouble(-0.1, 0.1);
                        individual.addPoint(Point2D(px, py));
                    }
                }
            } 
            else if (i < 2 * config.populationSize / 3) {
                // Triangular lattice initialization
                int radius = static_cast<int>(std::sqrt(numPoints));
                double sqrt3 = std::sqrt(3.0);
                for (int u = -radius; u <= radius && individual.size() < static_cast<size_t>(numPoints); ++u) {
                    for (int v = -radius; v <= radius && individual.size() < static_cast<size_t>(numPoints); ++v) {
                        double px = u * 1.0 + v * 0.5;
                        double py = v * sqrt3 / 2.0;
                        px += randomDouble(-0.1, 0.1);
                        py += randomDouble(-0.1, 0.1);
                        individual.addPoint(Point2D(px, py));
                    }
                }
            }
            else {
                // Random initialization
                for (int j = 0; j < numPoints; ++j) {
                    individual.addPoint(randomPoint());
                }
            }
            
            individual.evaluate();
            population.push_back(individual);
        }
        
        // Find best
        bestFitness = -1e9;
        for (auto& ind : population) {
            if (ind.getFitness() > bestFitness) {
                bestFitness = ind.getFitness();
                bestSolution = ind;
            }
        }
        
        generationsWithoutImprovement = 0;
    }
    
    PointSet tournamentSelection() {
        int bestIdx = rng() % config.populationSize;
        for (size_t i = 1; i < config.tournamentSize; ++i) {
            int candidate = rng() % config.populationSize;
            if (population[candidate].getFitness() > population[bestIdx].getFitness()) {
                bestIdx = candidate;
            }
        }
        return population[bestIdx];
    }
    
    void mutate(PointSet& individual) {
        for (size_t i = 0; i < individual.size(); ++i) {
            if (randomDouble() < config.mutationRate) {
                Point2D p = individual.getPoints()[i];
                
                if (randomDouble() < config.largeMutationRate) {
                    // Large mutation - random jump
                    p.x += randomDouble(-config.largeMutationStrength * config.searchBoundX,
                                        config.largeMutationStrength * config.searchBoundX);
                    p.y += randomDouble(-config.largeMutationStrength * config.searchBoundY,
                                        config.largeMutationStrength * config.searchBoundY);
                } else {
                    // Small mutation - Gaussian perturbation
                    p.x += normalDist(rng) * config.mutationStrength;
                    p.y += normalDist(rng) * config.mutationStrength;
                }
                
                // Clamp to bounds
                p.x = std::max(-config.searchBoundX, std::min(config.searchBoundX, p.x));
                p.y = std::max(-config.searchBoundY, std::min(config.searchBoundY, p.y));
                
                individual.setPoint(i, p);
            }
        }
        individual.evaluate();
    }
    
    PointSet crossover(const PointSet& a, const PointSet& b) {
        if (randomDouble() > config.crossoverRate) {
            return a;
        }
        
        PointSet child;
        child.targetDistance = config.targetDistance;
        child.distanceTolerance = config.distanceTolerance;
        child.minDistance = config.minDistance;
        
        size_t n = std::min(a.size(), b.size());
        
        for (size_t i = 0; i < n; ++i) {
            // Uniform crossover
            if (randomDouble() < 0.5) {
                child.addPoint(a.getPoints()[i]);
            } else {
                child.addPoint(b.getPoints()[i]);
            }
        }
        
        child.evaluate();
        return child;
    }
    
    void simulatedAnnealingRefinement(PointSet& solution, double temperature) {
        PointSet current = solution;
        double currentFitness = current.getFitness();
        
        for (int iter = 0; iter < config.iterationsPerTemp; ++iter) {
            PointSet candidate = current;
            
            // Perturb a random point
            int idx = rng() % candidate.size();
            Point2D p = candidate.getPoints()[idx];
            p.x += normalDist(rng) * temperature * 0.1;
            p.y += normalDist(rng) * temperature * 0.1;
            p.x = std::max(-config.searchBoundX, std::min(config.searchBoundX, p.x));
            p.y = std::max(-config.searchBoundY, std::min(config.searchBoundY, p.y));
            candidate.setPoint(idx, p);
            candidate.evaluate();
            
            double delta = candidate.getFitness() - currentFitness;
            
            if (delta > 0 || randomDouble() < std::exp(delta / temperature)) {
                current = candidate;
                currentFitness = candidate.getFitness();
                
                if (currentFitness > solution.getFitness()) {
                    solution = current;
                }
            }
        }
    }
    
    double computePopulationDiversity() const {
        if (population.empty()) return 0.0;
        
        double avgFitness = 0.0;
        for (const auto& ind : population) {
            avgFitness += ind.getFitness();
        }
        avgFitness /= population.size();
        
        double variance = 0.0;
        for (const auto& ind : population) {
            double diff = ind.getFitness() - avgFitness;
            variance += diff * diff;
        }
        
        return std::sqrt(variance / population.size());
    }
    
public:
    UnitDistanceOptimizer(const OptimizationConfig& cfg = OptimizationConfig()) 
        : config(cfg), rng(std::chrono::steady_clock::now().time_since_epoch().count()),
          uniformDist(0.0, 1.0), normalDist(0.0, 1.0), bestFitness(-1e9), 
          generationsWithoutImprovement(0) {}
    
    // Main optimization loop
    PointSet optimize(int numPoints) {
        std::cout << "Initializing population of " << config.populationSize 
                  << " individuals with " << numPoints << " points each...\n";
        initializePopulation(numPoints);
        
        std::cout << "Starting evolution for " << config.maxGenerations << " generations...\n";
        std::cout << "Target unit distance = " << config.targetDistance << "\n";
        std::cout << "Min separation = " << config.minDistance << "\n\n";
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        for (int gen = 0; gen < config.maxGenerations; ++gen) {
            // Create next generation
            std::vector<PointSet> nextGeneration;
            
            // Elitism
            std::sort(population.begin(), population.end(), 
                      [](const PointSet& a, const PointSet& b) { 
                          return a.getFitness() > b.getFitness(); 
                      });
            
            for (size_t i = 0; i < config.eliteCount; ++i) {
                nextGeneration.push_back(population[i]);
            }
            
            // Fill rest with crossover and mutation
            while (nextGeneration.size() < config.populationSize) {
                PointSet parent1 = tournamentSelection();
                PointSet parent2 = tournamentSelection();
                PointSet child = crossover(parent1, parent2);
                mutate(child);
                nextGeneration.push_back(child);
            }
            
            // Apply simulated annealing to best individuals
            double temperature = config.initialTemperature * 
                                 std::pow(config.coolingRate, gen);
            for (size_t i = 0; i < std::min(size_t(10), nextGeneration.size()); ++i) {
                simulatedAnnealingRefinement(nextGeneration[i], temperature);
            }
            
            population = std::move(nextGeneration);
            
            // Track best solution
            double currentBest = population[0].getFitness();
            if (currentBest > bestFitness) {
                bestFitness = currentBest;
                bestSolution = population[0];
                generationsWithoutImprovement = 0;
                
                std::cout << "Generation " << gen << ": Best fitness = " << bestFitness
                          << " (density = " << bestFitness / numPoints << ")\n";
            } else {
                generationsWithoutImprovement++;
            }
            
            // Record statistics
            fitnessHistory.push_back(bestFitness);
            diversityHistory.push_back(computePopulationDiversity());
            
            // Termination conditions
            if (config.targetFitness > 0 && bestFitness >= config.targetFitness) {
                std::cout << "\nTarget fitness reached!\n";
                break;
            }
            
            if (generationsWithoutImprovement >= config.stagnationLimit) {
                std::cout << "\nStagnation detected - stopping evolution\n";
                break;
            }
            
            // Progress indicator
            if (gen % 100 == 0 && gen > 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
                std::cout << "Progress: " << gen << "/" << config.maxGenerations 
                          << " gens, best=" << bestFitness 
                          << ", time=" << elapsed << "s\n";
            }
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
        
        std::cout << "\n=== Optimization Complete ===\n";
        std::cout << "Final best fitness: " << bestFitness << "\n";
        std::cout << "Unit distance density: " << bestFitness / numPoints << "\n";
        std::cout << "Total time: " << duration << " seconds\n";
        std::cout << "Generations: " << fitnessHistory.size() << "\n\n";
        
        return bestSolution;
    }
    
    // Access to results
    const PointSet& getBestSolution() const { return bestSolution; }
    double getBestFitness() const { return bestFitness; }
    const std::vector<double>& getFitnessHistory() const { return fitnessHistory; }
    const std::vector<double>& getDiversityHistory() const { return diversityHistory; }
    
    // Export results
    void exportResults(const std::string& filename) const {
        FILE* f = fopen(filename.c_str(), "w");
        if (!f) return;
        
        fprintf(f, "# Planar Unit Distance Optimization Results\n");
        fprintf(f, "# Points: %zu\n", bestSolution.size());
        fprintf(f, "# Unit distances: %zu\n", bestSolution.getEdges().size());
        fprintf(f, "# Density: %f\n", bestSolution.getFitness() / bestSolution.size());
        fprintf(f, "#\n# Point coordinates (x, y):\n");
        
        for (size_t i = 0; i < bestSolution.size(); ++i) {
            const auto& p = bestSolution.getPoints()[i];
            fprintf(f, "%zu\t%f\t%f\n", i, p.x, p.y);
        }
        
        fprintf(f, "#\n# Unit distance edges (i, j):\n");
        for (const auto& e : bestSolution.getEdges()) {
            fprintf(f, "%d\t%d\n", e.first, e.second);
        }
        
        fclose(f);
    }
    
    void updateConfig(const OptimizationConfig& cfg) { config = cfg; }
};

// ============================================================================
// Legacy Interface for Backward Compatibility
// ============================================================================

class UnitDistanceSolver {
private:
    double epsilon;
    std::vector<Point2D> points;
    std::unordered_set<Point2D, Point2DHash> pointSet;
    
public:
    UnitDistanceSolver(double eps = 1e-9) : epsilon(eps) {}
    
    // Construction methods
    UnitDistanceGraph constructSquareGrid(int n) {
        UnitDistanceGraph graph;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                graph.points.emplace_back(i, j);
            }
        }
        
        for (size_t i = 0; i < graph.points.size(); ++i) {
            for (size_t j = i + 1; j < graph.points.size(); ++j) {
                double dx = graph.points[i].x - graph.points[j].x;
                double dy = graph.points[i].y - graph.points[j].y;
                if (std::abs(dx*dx + dy*dy - 1.0) < epsilon) {
                    graph.edges.emplace_back(i, j);
                }
            }
        }
        return graph;
    }
    
    UnitDistanceGraph constructTriangularLattice(int radius) {
        UnitDistanceGraph graph;
        double sqrt3 = std::sqrt(3.0);
        
        for (int i = -radius; i <= radius; ++i) {
            for (int j = -radius; j <= radius; ++j) {
                double x = i * 1.0 + j * 0.5;
                double y = j * sqrt3 / 2.0;
                graph.points.emplace_back(x, y);
            }
        }
        
        for (size_t i = 0; i < graph.points.size(); ++i) {
            for (size_t j = i + 1; j < graph.points.size(); ++j) {
                double dx = graph.points[i].x - graph.points[j].x;
                double dy = graph.points[i].y - graph.points[j].y;
                if (std::abs(dx*dx + dy*dy - 1.0) < epsilon) {
                    graph.edges.emplace_back(i, j);
                }
            }
        }
        return graph;
    }
    
    // Optimized construction using evolutionary algorithm
    UnitDistanceGraph constructOptimized(int numPoints, const OptimizationConfig& cfg = OptimizationConfig()) {
        UnitDistanceOptimizer optimizer(cfg);
        PointSet best = optimizer.optimize(numPoints);
        
        UnitDistanceGraph graph;
        graph.points = best.getPoints();
        graph.edges = best.getEdges();
        
        return graph;
    }
    
    // OpenAI-inspired construction with local optimization
    UnitDistanceGraph constructOpenAIConstruction(int n) {
        // Use optimizer with conservative settings
        OptimizationConfig cfg;
        cfg.populationSize = 100;
        cfg.maxGenerations = 500;
        cfg.stagnationLimit = 100;
        cfg.mutationRate = 0.1;
        cfg.mutationStrength = 0.05;
        
        return constructOptimized(n, cfg);
    }
    
    // Analysis
    double computeDensity(const UnitDistanceGraph& graph) const {
        if (graph.points.empty()) return 0.0;
        return static_cast<double>(graph.edges.size()) / graph.points.size();
    }
    
    bool isLatticeArrangement(const UnitDistanceGraph& graph) const {
        if (graph.points.size() < 100) return false;
        
        std::set<std::pair<double, double>> differenceVectors;
        for (size_t i = 0; i < std::min(100UL, graph.points.size()); ++i) {
            for (size_t j = i + 1; j < std::min(100UL, graph.points.size()); ++j) {
                double dx = graph.points[j].x - graph.points[i].x;
                double dy = graph.points[j].y - graph.points[i].y;
                dx = std::round(dx * 1e6) / 1e6;
                dy = std::round(dy * 1e6) / 1e6;
                differenceVectors.insert({dx, dy});
            }
        }
        return differenceVectors.size() < graph.points.size() / 10;
    }
    
    bool validateGraph(const UnitDistanceGraph& graph) const {
        for (const auto& e : graph.edges) {
            double dx = graph.points[e.first].x - graph.points[e.second].x;
            double dy = graph.points[e.first].y - graph.points[e.second].y;
            if (std::abs(dx*dx + dy*dy - 1.0) > epsilon) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

double unitDistanceDensityUpperBound(int n) {
    return std::pow(n, 4.0/3.0);
}

double currentBestKnownDensity(int n) {
    return n * std::pow(std::log(n), 1.0/3.0);
}

} // namespace PlanarUnitDistance

#endif // PLANAR_UNIT_DISTANCE_HPP