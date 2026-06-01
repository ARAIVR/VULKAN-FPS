// PlanarUnitDistance.hpp
#ifndef PLANAR_UNIT_DISTANCE_HPP
#define PLANAR_UNIT_DISTANCE_HPP

#include <vector>
#include <set>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <functional>

namespace PlanarUnitDistance {

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
    std::vector<std::pair<int, int>> edges; // indices of points with unit distance
    
    int unitDistanceCount() const { return edges.size(); }
};

class UnitDistanceSolver {
private:
    double epsilon;
    std::vector<Point2D> points;
    std::unordered_set<Point2D, Point2DHash> pointSet;
    
    // Unit circle intersection helper
    std::vector<Point2D> circleIntersection(const Point2D& c1, double r1,
                                            const Point2D& c2, double r2) const;
    
    // Generate integer lattice points up to given bound
    std::vector<Point2D> generateLatticePoints(int bound) const;
    
    // Generate triangular lattice (hexagonal grid)
    std::vector<Point2D> generateTriangularLattice(int radius) const;
    
    // Check if two points have unit distance within epsilon
    bool isUnitDistance(const Point2D& a, const Point2D& b) const {
        double d2 = a.distanceSqTo(b);
        return std::abs(d2 - 1.0) < epsilon;
    }
    
public:
    UnitDistanceSolver(double eps = 1e-9) : epsilon(eps) {}
    
    // --- Construction methods for known point arrangements ---
    
    // Standard square grid N x N
    UnitDistanceGraph constructSquareGrid(int n);
    
    // Triangular / hexagonal lattice
    UnitDistanceGraph constructTriangularLattice(int radius);
    
    // "Moser spindle" - a unit distance graph requiring 4 colors
    UnitDistanceGraph constructMoserSpindle();
    
    // Golomb graph / Golomb ruler based arrangement
    UnitDistanceGraph constructGolombArrangement(const std::vector<int>& ruler);
    
    // Circle packing arrangement (points on a circle of radius R where chord length = 1)
    UnitDistanceGraph constructCircleArrangement(int n, double radius);
    
    // --- New construction discovered by OpenAI (simplified approximation) ---
    // Based on the reported breakthrough - a non-lattice arrangement
    // with higher unit distance density than square grids
    
    // Constructs the new best-known arrangement
    // Parameter 'n' controls number of points (should be composite)
    UnitDistanceGraph constructOpenAIConstruction(int n);
    
    // --- Analysis methods ---
    
    // Count total unit distances in a set of points
    UnitDistanceGraph analyzePointSet(const std::vector<Point2D>& pts);
    
    // Compute density: unit distances per point
    double computeDensity(const UnitDistanceGraph& graph) const;
    
    // Check if arrangement is a lattice (periodic)
    bool isLatticeArrangement(const UnitDistanceGraph& graph) const;
    
    // Find unit distance cliques
    std::vector<std::vector<int>> findUnitCliques(const UnitDistanceGraph& graph, int maxCliqueSize = 4);
    
    // --- Validation & verification ---
    
    // Verify if all stored edges are actually unit distance
    bool validateGraph(const UnitDistanceGraph& graph) const;
    
    // Get unique point count (should be same as points.size())
    size_t uniquePointCount() const { return pointSet.size(); }
};

// Additional utility functions
double unitDistanceDensityUpperBound(int n);
double currentBestKnownDensity(int n);

} // namespace PlanarUnitDistance

#endif // PLANAR_UNIT_DISTANCE_HPP