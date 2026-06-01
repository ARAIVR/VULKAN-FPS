// PlanarUnitDistance.cpp
#include "PlanarUnitDistance.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

namespace PlanarUnitDistance {

// Additional implementation for any missing methods
// Most methods are implemented inline in the header for performance

// Helper function to export results in various formats
void exportToCSV(const PointSet& solution, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "id,x,y\n";
    for (size_t i = 0; i < solution.size(); ++i) {
        const auto& p = solution.getPoints()[i];
        file << i << "," << p.x << "," << p.y << "\n";
    }
    
    file.close();
}

void exportEdgesToCSV(const PointSet& solution, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "from,to\n";
    for (const auto& e : solution.getEdges()) {
        file << e.first << "," << e.second << "\n";
    }
    
    file.close();
}

} // namespace PlanarUnitDistance