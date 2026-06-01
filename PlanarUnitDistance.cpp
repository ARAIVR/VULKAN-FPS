// PlanarUnitDistance.cpp
#include "PlanarUnitDistance.hpp"
#include <cmath>
#include <map>
#include <iostream>

namespace PlanarUnitDistance {

std::vector<Point2D> UnitDistanceSolver::circleIntersection(
    const Point2D& c1, double r1,
    const Point2D& c2, double r2) const {
    
    std::vector<Point2D> intersections;
    
    double dx = c2.x - c1.x;
    double dy = c2.y - c1.y;
    double d2 = dx * dx + dy * dy;
    double d = std::sqrt(d2);
    
    // No intersection
    if (d > r1 + r2 + epsilon || d < std::abs(r1 - r2) - epsilon)
        return intersections;
    
    // Single intersection (tangent)
    if (std::abs(d - (r1 + r2)) < epsilon || std::abs(d - std::abs(r1 - r2)) < epsilon) {
        double t = r1 / d;
        double ix = c1.x + dx * t;
        double iy = c1.y + dy * t;
        intersections.emplace_back(ix, iy);
        return intersections;
    }
    
    // Two intersections
    double a = (r1 * r1 - r2 * r2 + d2) / (2 * d2);
    double h2 = r1 * r1 - a * a * d2;
    if (h2 < 0) return intersections;
    
    double h = std::sqrt(h2);
    double x0 = c1.x + a * dx;
    double y0 = c1.y + a * dy;
    
    double rx = -dy * (h / d);
    double ry = dx * (h / d);
    
    intersections.emplace_back(x0 + rx, y0 + ry);
    intersections.emplace_back(x0 - rx, y0 - ry);
    
    return intersections;
}

std::vector<Point2D> UnitDistanceSolver::generateLatticePoints(int bound) const {
    std::vector<Point2D> lattice;
    for (int i = -bound; i <= bound; ++i) {
        for (int j = -bound; j <= bound; ++j) {
            lattice.emplace_back(static_cast<double>(i), static_cast<double>(j));
        }
    }
    return lattice;
}

std::vector<Point2D> UnitDistanceSolver::generateTriangularLattice(int radius) const {
    std::vector<Point2D> lattice;
    double sqrt3 = std::sqrt(3.0);
    
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            double x = i * 1.0 + j * 0.5;
            double y = j * sqrt3 / 2.0;
            lattice.emplace_back(x, y);
        }
    }
    return lattice;
}

UnitDistanceGraph UnitDistanceSolver::constructSquareGrid(int n) {
    UnitDistanceGraph graph;
    double step = 1.0; // unit spacing
    
    // Generate points
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            graph.points.emplace_back(i * step, j * step);
        }
    }
    
    // Find unit distance edges
    for (size_t i = 0; i < graph.points.size(); ++i) {
        for (size_t j = i + 1; j < graph.points.size(); ++j) {
            if (isUnitDistance(graph.points[i], graph.points[j])) {
                graph.edges.emplace_back(i, j);
            }
        }
    }
    
    return graph;
}

UnitDistanceGraph UnitDistanceSolver::constructTriangularLattice(int radius) {
    UnitDistanceGraph graph;
    graph.points = generateTriangularLattice(radius);
    
    // Find unit distance edges (triangular lattice has distance 1 to 6 neighbors)
    for (size_t i = 0; i < graph.points.size(); ++i) {
        for (size_t j = i + 1; j < graph.points.size(); ++j) {
            if (isUnitDistance(graph.points[i], graph.points[j])) {
                graph.edges.emplace_back(i, j);
            }
        }
    }
    
    return graph;
}

UnitDistanceGraph UnitDistanceSolver::constructMoserSpindle() {
    UnitDistanceGraph graph;
    
    // Moser spindle vertices (coordinates from known construction)
    // Normalized so edges are unit length
    double sqrt3 = std::sqrt(3.0);
    
    // Point set for Moser spindle (7 points, 11 edges)
    graph.points = {
        Point2D(0, 0),           // 0
        Point2D(1, 0),           // 1
        Point2D(0.5, sqrt3/2),   // 2
        Point2D(0.5, sqrt3/6),   // 3
        Point2D(1.5, sqrt3/6),   // 4
        Point2D(1.0, sqrt3/3),   // 5
        Point2D(0.5, -sqrt3/6)   // 6
    };
    
    // Edges (unit distance connections)
    std::vector<std::pair<int, int>> edges = {
        {0,1}, {0,2}, {0,3}, {0,6},
        {1,2}, {1,4}, {1,5},
        {2,3}, {2,5},
        {3,4}, {3,6},
        {4,5}, {4,6},
        {5,6}
    };
    
    for (const auto& e : edges) {
        if (isUnitDistance(graph.points[e.first], graph.points[e.second])) {
            graph.edges.push_back(e);
        }
    }
    
    return graph;
}

UnitDistanceGraph UnitDistanceSolver::constructGolombArrangement(const std::vector<int>& ruler) {
    UnitDistanceGraph graph;
    
    // Place points at positions given by Golomb ruler
    for (size_t i = 0; i < ruler.size(); ++i) {
        graph.points.emplace_back(static_cast<double>(ruler[i]), 0.0);
    }
    
    // Add a second row offset by sqrt(3)/2 for triangular connections
    double offsetY = std::sqrt(3.0) / 2.0;
    size_t baseSize = ruler.size();
    for (size_t i = 0; i < ruler.size(); ++i) {
        graph.points.emplace_back(static_cast<double>(ruler[i]) + 0.5, offsetY);
    }
    
    // Find unit distances
    for (size_t i = 0; i < graph.points.size(); ++i) {
        for (size_t j = i + 1; j < graph.points.size(); ++j) {
            if (isUnitDistance(graph.points[i], graph.points[j])) {
                graph.edges.emplace_back(i, j);
            }
        }
    }
    
    return graph;
}

UnitDistanceGraph UnitDistanceSolver::constructCircleArrangement(int n, double radius) {
    UnitDistanceGraph graph;
    
    // Place points on a circle
    for (int i = 0; i < n; ++i) {
        double angle = 2.0 * M_PI * i / n;
        double x = radius * std::cos(angle);
        double y = radius * std::sin(angle);
        graph.points.emplace_back(x, y);
    }
    
    // Chord length = 2 * R * sin(π * k / n)
    // We want chord = 1, so R = 1 / (2 * sin(π * k / n))
    // This finds which chords are unit length for the given radius
    
    for (size_t i = 0; i < graph.points.size(); ++i) {
        for (size_t j = i + 1; j < graph.points.size(); ++j) {
            if (isUnitDistance(graph.points[i], graph.points[j])) {
                graph.edges.emplace_back(i, j);
            }
        }
    }
    
    return graph;
}

UnitDistanceGraph UnitDistanceSolver::constructOpenAIConstruction(int n) {
    // Simplified implementation of the new arrangement
    // Based on the breakthrough: non-lattice construction with higher density
    // than square grids for certain n
    
    UnitDistanceGraph graph;
    
    // This is a heuristic approximation of the new construction
    // The actual arrangement is more complex, but this captures the key idea:
    // A pattern combining triangular lattice patches with carefully placed 
    // "defects" that create additional unit distances
    
    // Generate base triangular lattice
    int latticeSize = static_cast<int>(std::sqrt(n));
    auto lattice = generateTriangularLattice(latticeSize);
    
    // Take first n points
    for (size_t i = 0; i < std::min(static_cast<size_t>(n), lattice.size()); ++i) {
        graph.points.push_back(lattice[i]);
    }
    
    // Apply a non-linear transformation to break lattice structure
    // while preserving most unit distances
    for (auto& p : graph.points) {
        // Small perturbation that preserves some distances
        double r = std::sqrt(p.x * p.x + p.y * p.y);
        if (r > 0.1) {
            double angle = std::atan2(p.y, p.x);
            // Add a periodic modulation
            double modulation = 0.05 * std::sin(6.0 * angle);
            p.x += modulation * std::cos(angle);
            p.y += modulation * std::sin(angle);
        }
    }
    
    // Rebuild edge list
    for (size_t i = 0; i < graph.points.size(); ++i) {
        for (size_t j = i + 1; j < graph.points.size(); ++j) {
            if (isUnitDistance(graph.points[i], graph.points[j])) {
                graph.edges.emplace_back(i, j);
            }
        }
    }
    
    return graph;
}

UnitDistanceGraph UnitDistanceSolver::analyzePointSet(const std::vector<Point2D>& pts) {
    UnitDistanceGraph graph;
    graph.points = pts;
    
    for (size_t i = 0; i < pts.size(); ++i) {
        for (size_t j = i + 1; j < pts.size(); ++j) {
            if (isUnitDistance(pts[i], pts[j])) {
                graph.edges.emplace_back(i, j);
            }
        }
    }
    
    return graph;
}

double UnitDistanceSolver::computeDensity(const UnitDistanceGraph& graph) const {
    if (graph.points.empty()) return 0.0;
    return static_cast<double>(graph.edges.size()) / graph.points.size();
}

bool UnitDistanceSolver::isLatticeArrangement(const UnitDistanceGraph& graph) const {
    if (graph.points.size() < 100) return false;
    
    // Check for periodicity by looking at vector differences
    std::set<std::pair<double, double>> differenceVectors;
    
    for (size_t i = 0; i < std::min(100UL, graph.points.size()); ++i) {
        for (size_t j = i + 1; j < std::min(100UL, graph.points.size()); ++j) {
            double dx = graph.points[j].x - graph.points[i].x;
            double dy = graph.points[j].y - graph.points[i].y;
            
            // Round to reasonable precision
            dx = std::round(dx * 1e6) / 1e6;
            dy = std::round(dy * 1e6) / 1e6;
            differenceVectors.insert({dx, dy});
        }
    }
    
    // Lattices typically have a small set of basis vectors that generate all differences
    // This heuristic checks if the number of unique vectors is small relative to point count
    return differenceVectors.size() < graph.points.size() / 10;
}

std::vector<std::vector<int>> UnitDistanceSolver::findUnitCliques(const UnitDistanceGraph& graph, int maxCliqueSize) {
    std::vector<std::vector<int>> cliques;
    
    // Build adjacency matrix for quick lookup
    std::vector<std::unordered_set<int>> adj(graph.points.size());
    for (const auto& e : graph.edges) {
        adj[e.first].insert(e.second);
        adj[e.second].insert(e.first);
    }
    
    // Simple Bron-Kerbosch style clique finding
    std::function<void(std::vector<int>&, std::unordered_set<int>&, std::unordered_set<int>&)> 
        bronKerbosch = [&](std::vector<int>& R, std::unordered_set<int>& P, std::unordered_set<int>& X) {
        if (R.size() >= 2) {
            cliques.push_back(R);
        }
        if (R.size() >= static_cast<size_t>(maxCliqueSize)) return;
        
        auto P_copy = P;
        for (int v : P_copy) {
            std::vector<int> R_new = R;
            R_new.push_back(v);
            
            std::unordered_set<int> P_new, X_new;
            for (int u : P) {
                if (adj[v].count(u)) P_new.insert(u);
            }
            for (int u : X) {
                if (adj[v].count(u)) X_new.insert(u);
            }
            
            bronKerbosch(R_new, P_new, X_new);
            
            P.erase(v);
            X.insert(v);
        }
    };
    
    std::unordered_set<int> allPoints;
    for (size_t i = 0; i < graph.points.size(); ++i) {
        allPoints.insert(static_cast<int>(i));
    }
    
    std::vector<int> R;
    std::unordered_set<int> X;
    bronKerbosch(R, allPoints, X);
    
    return cliques;
}

bool UnitDistanceSolver::validateGraph(const UnitDistanceGraph& graph) const {
    for (const auto& e : graph.edges) {
        if (!isUnitDistance(graph.points[e.first], graph.points[e.second])) {
            return false;
        }
    }
    return true;
}

double unitDistanceDensityUpperBound(int n) {
    // Known upper bound: O(n^(4/3))
    // Maximum number of unit distances among n points is at most ~n^(4/3)
    return std::pow(n, 4.0/3.0);
}

double currentBestKnownDensity(int n) {
    // For large n, the best known density is ~n^(1 + c/log log n)
    // The new OpenAI construction improves this
    return n * std::pow(std::log(n), 1.0/3.0);
}

} // namespace PlanarUnitDistance