#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <tuple>
#include <vector>

// Computes interior geodesic distances between source and cage vertices by
// voxelizing the cage mesh and running Dijkstra on the cost volume
// (cost 1.0 inside the cage, 1e10 outside). Paths must stay inside the cage,
// so distances respect the interior topology rather than straight-line distance.
class InteriorGeodesicDistance
{
public:
    // Returns an (N_source x N_cage) matrix of world-space interior geodesic
    // distances. Values may be very large for pairs with no interior path.
    // targetVoxelResolution controls the longest grid dimension (default 80).
    static Eigen::MatrixXf Compute(
        const Eigen::MatrixXd& cageVerts,
        const Eigen::MatrixXi& cageFaces,
        const Eigen::MatrixXd& sourceVerts,
        int targetVoxelResolution = 80);

private:
    struct Grid {
        std::vector<float> cost;  // index = ix + iy*nx + iz*nx*ny; 1.0=inside, 1e10=outside
        int nx = 0, ny = 0, nz = 0;
        double ox = 0, oy = 0, oz = 0;
        double pitch = 1.0;

        int  Idx(int ix, int iy, int iz) const { return ix + iy * nx + iz * nx * ny; }
        bool Valid(int ix, int iy, int iz) const {
            return ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz;
        }
        Eigen::Vector3i WorldToIdx(const Eigen::Vector3d& p) const {
            return {
                static_cast<int>((p.x() - ox) / pitch),
                static_cast<int>((p.y() - oy) / pitch),
                static_cast<int>((p.z() - oz) / pitch)
            };
        }
        Eigen::Vector3i ClampIdx(Eigen::Vector3i idx) const {
            idx.x() = std::max(0, std::min(nx - 1, idx.x()));
            idx.y() = std::max(0, std::min(ny - 1, idx.y()));
            idx.z() = std::max(0, std::min(nz - 1, idx.z()));
            return idx;
        }
    };

    static Grid BuildGrid(
        const Eigen::MatrixXd& verts,
        const Eigen::MatrixXi& faces,
        int targetRes);

    // Returns x-intersection of triangle (v0,v1,v2) with the horizontal ray
    // at (y=cy, z=cz) in the +X direction. Returns false if no intersection.
    static bool TriangleScanlineX(
        double cy, double cz,
        const Eigen::RowVector3d& v0,
        const Eigen::RowVector3d& v1,
        const Eigen::RowVector3d& v2,
        double& xOut);

    // Dijkstra from seed voxel; returns per-voxel world-space distances.
    static std::vector<float> Dijkstra3D(const Grid& g, int sx, int sy, int sz);
};

// ----- implementation -----

inline Eigen::MatrixXf InteriorGeodesicDistance::Compute(
    const Eigen::MatrixXd& cageVerts,
    const Eigen::MatrixXi& cageFaces,
    const Eigen::MatrixXd& sourceVerts,
    int targetVoxelResolution)
{
    const int N_cage   = static_cast<int>(cageVerts.rows());
    const int N_source = static_cast<int>(sourceVerts.rows());

    Eigen::MatrixXf dist = Eigen::MatrixXf::Constant(
        N_source, N_cage, std::numeric_limits<float>::quiet_NaN());

    if (N_cage == 0 || N_source == 0 || cageFaces.rows() == 0)
        return dist;

    const Grid g = BuildGrid(cageVerts, cageFaces, targetVoxelResolution);

    // Map cage vertices to clamped voxel indices
    std::vector<Eigen::Vector3i> cageIdx(N_cage);
    for (int j = 0; j < N_cage; ++j)
        cageIdx[j] = g.ClampIdx(g.WorldToIdx(cageVerts.row(j).transpose()));

    // Group source vertices by their voxel to deduplicate Dijkstra runs
    std::map<std::tuple<int,int,int>, std::vector<int>> voxelToSources;
    for (int i = 0; i < N_source; ++i) {
        Eigen::Vector3i idx = g.ClampIdx(g.WorldToIdx(sourceVerts.row(i).transpose()));
        voxelToSources[{idx.x(), idx.y(), idx.z()}].push_back(i);
    }

    for (auto& [key, srcIndices] : voxelToSources) {
        auto [sx, sy, sz] = key;
        const std::vector<float> distVol = Dijkstra3D(g, sx, sy, sz);

        for (int j = 0; j < N_cage; ++j) {
            const auto& ci = cageIdx[j];
            const float d = distVol[g.Idx(ci.x(), ci.y(), ci.z())];
            for (int si : srcIndices)
                dist(si, j) = d;
        }
    }
    return dist;
}

inline InteriorGeodesicDistance::Grid InteriorGeodesicDistance::BuildGrid(
    const Eigen::MatrixXd& verts,
    const Eigen::MatrixXi& faces,
    int targetRes)
{
    Grid g;
    const double padding = 1e-3;
    Eigen::Vector3d mn = verts.colwise().minCoeff();
    Eigen::Vector3d mx = verts.colwise().maxCoeff();
    mn.array() -= padding;
    mx.array() += padding;

    const Eigen::Vector3d extent = mx - mn;
    double maxExtent = extent.maxCoeff();
    if (maxExtent < 1e-12) maxExtent = 1.0;

    g.pitch = maxExtent / static_cast<double>(targetRes);
    g.ox = mn.x(); g.oy = mn.y(); g.oz = mn.z();
    g.nx = static_cast<int>(std::ceil(extent.x() / g.pitch)) + 1;
    g.ny = static_cast<int>(std::ceil(extent.y() / g.pitch)) + 1;
    g.nz = static_cast<int>(std::ceil(extent.z() / g.pitch)) + 1;
    g.cost.assign(static_cast<size_t>(g.nx) * g.ny * g.nz, 1e10f);

    // Scanline voxelization: for each (iy, iz) cast a ray in +X and collect
    // x-crossings with cage triangles; voxels between crossing pairs are inside.
    for (int iz = 0; iz < g.nz; ++iz) {
        const double cz = g.oz + (iz + 0.5) * g.pitch;
        for (int iy = 0; iy < g.ny; ++iy) {
            const double cy = g.oy + (iy + 0.5) * g.pitch;

            std::vector<double> xs;
            for (int f = 0; f < faces.rows(); ++f) {
                double xHit;
                if (TriangleScanlineX(cy, cz,
                        verts.row(faces(f, 0)),
                        verts.row(faces(f, 1)),
                        verts.row(faces(f, 2)), xHit))
                    xs.push_back(xHit);
            }
            std::sort(xs.begin(), xs.end());

            for (size_t k = 0; k + 1 < xs.size(); k += 2) {
                const int ix0 = std::max(0, static_cast<int>((xs[k]     - g.ox) / g.pitch));
                const int ix1 = std::min(g.nx - 1, static_cast<int>((xs[k + 1] - g.ox) / g.pitch));
                for (int ix = ix0; ix <= ix1; ++ix)
                    g.cost[g.Idx(ix, iy, iz)] = 1.0f;
            }
        }
    }
    return g;
}

inline bool InteriorGeodesicDistance::TriangleScanlineX(
    double cy, double cz,
    const Eigen::RowVector3d& v0,
    const Eigen::RowVector3d& v1,
    const Eigen::RowVector3d& v2,
    double& xOut)
{
    // Solve (1-s-t)*v0 + s*v1 + t*v2 = (?, cy, cz) for barycentric (s,t).
    const double ay = v1.y() - v0.y(), az = v1.z() - v0.z();
    const double by = v2.y() - v0.y(), bz = v2.z() - v0.z();
    const double det = ay * bz - az * by;
    if (std::abs(det) < 1e-12) return false;

    const double ry = cy - v0.y(), rz = cz - v0.z();
    const double s = (ry * bz - rz * by) / det;
    const double t = (ay * rz - az * ry) / det;
    constexpr double eps = 1e-7;
    if (s < -eps || t < -eps || s + t > 1.0 + eps) return false;

    xOut = v0.x() + s * (v1.x() - v0.x()) + t * (v2.x() - v0.x());
    return true;
}

inline std::vector<float> InteriorGeodesicDistance::Dijkstra3D(
    const Grid& g, int sx, int sy, int sz)
{
    const int N = g.nx * g.ny * g.nz;
    std::vector<float> dist(N, std::numeric_limits<float>::infinity());

    // 26-connected neighborhood offsets and precomputed Euclidean step lengths
    static constexpr int DX[26] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };
    static constexpr int DY[26] = {
        -1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 0, 1, 1, 1
    };
    static constexpr int DZ[26] = {
        -1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1
    };
    static constexpr float STEP[26] = {
        1.7320508f, 1.4142136f, 1.7320508f,
        1.4142136f, 1.0f,       1.4142136f,
        1.7320508f, 1.4142136f, 1.7320508f,
        1.4142136f, 1.0f,       1.4142136f,
        1.0f,                   1.0f,
        1.4142136f, 1.0f,       1.4142136f,
        1.7320508f, 1.4142136f, 1.7320508f,
        1.4142136f, 1.0f,       1.4142136f,
        1.7320508f, 1.4142136f, 1.7320508f
    };

    const int seed = g.Idx(sx, sy, sz);
    dist[seed] = 0.0f;

    using Entry = std::pair<float, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
    pq.push({0.0f, seed});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;

        const int iz  = u / (g.nx * g.ny);
        const int rem = u % (g.nx * g.ny);
        const int iy  = rem / g.nx;
        const int ix  = rem % g.nx;
        const float cu = g.cost[u];

        for (int k = 0; k < 26; ++k) {
            const int nx_ = ix + DX[k];
            const int ny_ = iy + DY[k];
            const int nz_ = iz + DZ[k];
            if (!g.Valid(nx_, ny_, nz_)) continue;

            const int   v       = g.Idx(nx_, ny_, nz_);
            const float cv      = g.cost[v];
            const float edgeCost = 0.5f * (cu + cv) * STEP[k] * static_cast<float>(g.pitch);
            const float nd      = dist[u] + edgeCost;
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }
    return dist;
}
