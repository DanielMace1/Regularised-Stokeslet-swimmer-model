import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from typing import Protocol, List, Optional, Callable
import time

class Component(Protocol):
    def body_points(self, t: float) -> np.ndarray:   # (Ni,3)
        ...
    def body_u_shape(self, t: float) -> np.ndarray:  # (Ni,3)
        ...
    def weights(self, t: float) -> np.ndarray:       # (Ni,)
        ...
@dataclass
class CompositeSwimmer:
    components: List[Component]

    def body_state(self,t:float):
        pts_list, ush_list, w_list = [], [], []
        for comp in self.components:
            p = comp.body_points(t)
            u = comp.body_u_shape(t)
            w = comp.weights(t)
            if p.shape != u.shape:
                raise ValueError("body_points and body_u_shape must have same shape")
            if w.shape != (p.shape[0],):
                raise ValueError("weights must have shape (Ni,)")
            pts_list.append(p)
            ush_list.append(u)
            w_list.append(w)
        pts = np.vstack(pts_list)
        ush = np.vstack(ush_list)
        w = np.concatenate(w_list)
        return pts, ush, w

class SinusoidalWave:
    def __init__(self, a, k, omega):
        self.a = a
        self.k = k
        self.omega = omega

    def r(self, s, t):
        phase = self.k*s - self.omega*t
        x = s
        y = self.a * np.sin(phase)
        z = np.zeros_like(s)
        pts = np.vstack((x, y, z)).T
        pts = pts - pts[0]
        return pts

    def r_t(self, s, t):
        phase = self.k*s - self.omega*t
        x_t = np.zeros_like(s)
        y_t = -self.a * self.omega * np.cos(phase)
        z_t = np.zeros_like(s)
        u =np.vstack((x_t, y_t, z_t)).T
        u = u - u[0]
        return u


class SinusoidalWaveCutoff:
    def __init__(self, a, k, omega, L=1.0, n_fine=5000, phi = 0.0):
        self.a = a
        self.k = k
        self.omega = omega
        self.L = L
        self.n_fine = n_fine
        self.phi = phi

    def fine_curve(self, x, t):
        phase = self.k * x - self.omega * t + self.phi
        A = self.a*(0.1087*x + 0.0543)*(1 - np.exp(-64*x**2))
        y = A * np.sin(phase)
        z = np.zeros_like(x)
        return np.column_stack((x, y, z))

    def build_curve_to_length(self, t):
        if hasattr(self, '_cache_t') and self._cache_t == t:
            return self._cache_Xcut, self._cache_Scut
        x_max = self.L
        x = np.linspace(0.0, x_max, self.n_fine)
        X = self.fine_curve(x, t)

        seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
        S = np.concatenate(([0.0], np.cumsum(seg)))

        while S[-1] < self.L:
            x_max *= 1.5
            x = np.linspace(0.0, x_max, self.n_fine)
            X = self.fine_curve(x, t)
            seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
            S = np.concatenate(([0.0], np.cumsum(seg)))

        # cut off exactly at arclength L
        idx = np.searchsorted(S, self.L)

        if S[idx] == self.L:
            Xcut = X[:idx+1]
            Scut = S[:idx+1]
        else:
            lam = (self.L - S[idx-1]) / (S[idx] - S[idx-1])
            tip = (1.0 - lam) * X[idx-1] + lam * X[idx]

            Xcut = np.vstack((X[:idx], tip))
            Scut = np.concatenate((S[:idx], [self.L]))
        self._cache_t = t
        self._cache_Xcut = Xcut
        self._cache_Scut = Scut
        return Xcut, Scut

    def r(self, s, t):
        Xcut, Scut = self.build_curve_to_length(t)
        x = np.interp(s, Scut, Xcut[:, 0])
        y = np.interp(s, Scut, Xcut[:, 1])
        z = np.interp(s, Scut, Xcut[:, 2])
        pts = np.vstack((x, y, z)).T
        return pts

    def r_t(self, s, t,dt = 1e-5):
        u = (self.r(s, t+dt) - self.r(s, t-dt)) / (2*dt)
        return u

class HumanWave:
    def __init__(self, a, k, omega):
        self.a = a
        self.k = k
        self.omega = omega

    def r(self, s, t):
        phase = self.k*s - self.omega*t
        x = s
        y = (self.a*s+self.a/2) * np.sin(phase)
        z = np.zeros_like(s)
        return np.vstack((x, y,z)).T

    def r_t(self, s, t):
        phase = self.k*s - self.omega*t
        x_t = np.zeros_like(s)
        y_t = -(self.a*s+self.a/2) * self.omega * np.cos(phase)
        z_t = np.zeros_like(s)
        return np.vstack((x_t, y_t, z_t)).T

class HelicalWaveCutoff:
    def __init__(self, a, k, omega, L=1.0, n_fine=5000, phi=0.0):
        self.a = a
        self.k = k
        self.omega = omega
        self.L = L
        self.n_fine = n_fine
        self.phi = phi

    def fine_curve(self, x, t):
        phase = self.k * x - self.omega * t + self.phi
        A = self.a*(0.1087*x + 0.0543)*(1 - np.exp(-64*x**2))
        y = A * np.cos(phase)
        z = A * np.sin(phase)
        return np.column_stack((x, y, z))

    def build_curve_to_length(self, t):
        if hasattr(self, '_cache_t') and self._cache_t == t:
            return self._cache_Xcut, self._cache_Scut
        x_max = self.L
        x = np.linspace(0.0, x_max, self.n_fine)
        X = self.fine_curve(x, t)

        seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
        S = np.concatenate(([0.0], np.cumsum(seg)))

        while S[-1] < self.L:
            x_max *= 1.5
            x = np.linspace(0.0, x_max, self.n_fine)
            X = self.fine_curve(x, t)
            seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
            S = np.concatenate(([0.0], np.cumsum(seg)))

        idx = np.searchsorted(S, self.L)
        if S[idx] == self.L:
            Xcut = X[:idx+1]
            Scut = S[:idx+1]
        else:
            lam = (self.L - S[idx-1]) / (S[idx] - S[idx-1])
            tip = (1.0 - lam) * X[idx-1] + lam * X[idx]
            Xcut = np.vstack((X[:idx], tip))
            Scut = np.concatenate((S[:idx], [self.L]))

        self._cache_t = t
        self._cache_Xcut = Xcut
        self._cache_Scut = Scut
        return Xcut, Scut

    def r(self, s, t):
        Xcut, Scut = self.build_curve_to_length(t)
        x = np.interp(s, Scut, Xcut[:, 0])
        y = np.interp(s, Scut, Xcut[:, 1])
        z = np.interp(s, Scut, Xcut[:, 2])
        return np.vstack((x, y, z)).T

    def r_t(self, s, t, dt=1e-5):
        return (self.r(s, t+dt) - self.r(s, t-dt)) / (2*dt)

class EllipticalWaveCutoff:
    def __init__(self, ay, az, k, omega, L=1.0, n_fine=5000, phi=0.0):
        self.ay = ay
        self.az = az
        self.k = k
        self.omega = omega
        self.L = L
        self.n_fine = n_fine
        self.phi = phi

    def fine_curve(self, x, t):
        phase = self.k * x - self.omega * t + self.phi
        A = (0.1087*x + 0.0543)*(1 - np.exp(-64*x**2))
        y = self.ay * A * np.cos(phase)
        z = self.az * A * np.sin(phase)
        return np.column_stack((x, y, z))

    def build_curve_to_length(self, t):
        if hasattr(self, '_cache_t') and self._cache_t == t:
            return self._cache_Xcut, self._cache_Scut
        x_max = self.L
        x = np.linspace(0.0, x_max, self.n_fine)
        X = self.fine_curve(x, t)
        seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
        S = np.concatenate(([0.0], np.cumsum(seg)))
        while S[-1] < self.L:
            x_max *= 1.5
            x = np.linspace(0.0, x_max, self.n_fine)
            X = self.fine_curve(x, t)
            seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
            S = np.concatenate(([0.0], np.cumsum(seg)))
        idx = np.searchsorted(S, self.L)
        if S[idx] == self.L:
            Xcut = X[:idx+1]
            Scut = S[:idx+1]
        else:
            lam = (self.L - S[idx-1]) / (S[idx] - S[idx-1])
            tip = (1.0 - lam) * X[idx-1] + lam * X[idx]
            Xcut = np.vstack((X[:idx], tip))
            Scut = np.concatenate((S[:idx], [self.L]))
        self._cache_t = t
        self._cache_Xcut = Xcut
        self._cache_Scut = Scut
        return Xcut, Scut

    def r(self, s, t):
        Xcut, Scut = self.build_curve_to_length(t)
        x = np.interp(s, Scut, Xcut[:, 0])
        y = np.interp(s, Scut, Xcut[:, 1])
        z = np.interp(s, Scut, Xcut[:, 2])
        return np.vstack((x, y, z)).T

    def r_t(self, s, t, dt=1e-5):
        return (self.r(s, t+dt) - self.r(s, t-dt)) / (2*dt)

class HelicalWave:
    # rshape(s, t) = (s, a*cos(ks - wt), a*sin(ks - wt))
    def __init__(self, a, k, omega):
        self.a = a
        self.k = k
        self.omega = omega

    def r(self, s, t):
        phase = self.k*s - self.omega*t
        x = s
        y = self.a * np.cos(phase)
        z = self.a * np.sin(phase)
        return np.vstack((x, y,z)).T

    def r_t(self, s, t):
        phase = self.k*s - self.omega*t
        x_t = np.zeros_like(s)
        y_t = self.a * self.omega * np.sin(phase)
        z_t = -self.a * self.omega * np.cos(phase)
        return np.vstack((x_t, y_t, z_t)).T


class Elliptical3DWave:
    def __init__(self, ay, az, k, omega, phi=0.0):
        self.ay, self.az, self.k, self.omega, self.phi = ay, az, k, omega, phi

    def r(self, s, t):
        phase = self.k*s - self.omega*t
        return np.vstack((s,
                          self.ay*np.cos(phase),
                          self.az*np.sin(phase + self.phi))).T

    def r_t(self, s, t):
        phase = self.k*s - self.omega*t
        return np.vstack((np.zeros_like(s),
                          self.ay*self.omega*np.sin(phase),
                          -self.az*self.omega*np.cos(phase + self.phi))).T

def K_mat(r, eps, mu):
    R = np.sqrt(np.dot(r,r) + eps**2)
    K = ((1.0/R+eps**2/R**3)*np.eye(3) + np.outer(r,r)/R**3)/(8.0*np.pi*mu)
    return K

def crossMat(r):
    rx, ry, rz = r
    return np.array([[0, -rz,  ry],
                     [rz,  0, -rx],
                     [-ry, rx, 0]], dtype=float)

def M_mat(points,eps,mu, w=None):
    points = np.asarray(points,dtype = float)
    N = points.shape[0]

    if w is None:
        w = np.ones(N,dtype = float)
    else:
        w = np.asarray(w, float)
        if w.shape != (N,):
            raise ValueError("w must have shape (N,)")

    M = np.zeros((3*N,3*N),dtype=float)
    for i in range(N):
        ri = points[i]
        row = slice(3*i,3*i+3)
        for j in range(N):
            r = ri - points[j]
            K = K_mat(r, eps, mu)
            col = slice(3*j,3*j+3)
            M[row,col] = K * w[j] #quadrature weight
    return M

def stack_velocities(v):
    return np.asarray(v,float).reshape(-1)

def buildR_U(swimmer_id,Ntot,S): #identity matrix R_U
    R_U = np.zeros((3*Ntot, 3*S),float)
    for i in range(Ntot):
        s = swimmer_id[i]
        R_U[3*i:3*i+3, 3*s:3*s+3] = np.eye(3)
    return R_U

def buildR_Omega(points,swimmer_id,x0s,Ntot,S): #cross product matrix R_Omega
    R_Omega = np.zeros((3*Ntot, 3*S),float)
    for i in range(Ntot):
        s = swimmer_id[i]
        r = points[i] - x0s[s]
        R_Omega[3*i:3*i+3, 3*s:3*s+3] = -crossMat(r)
    return R_Omega

def rot_from_omega(Omega, dt):
    w = np.asarray(Omega, float)
    norm_w = np.linalg.norm(w)
    theta = norm_w * dt
    if theta < 1e-12:
        return np.eye(3)
    k = w / norm_w
    K = crossMat(k)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)


def build_constraint_mat(points,swimmer_id,x0s,Ntot,S,w=None): #CF and CT for force and torque constraints
    if w is None:
        w = np.ones(Ntot,dtype = float)
    else:
        w = np.asarray(w, float)
        if w.shape != (Ntot,):
            raise ValueError("w must have shape (N,)")
    
    CF = np.zeros((3*S,3*Ntot),float)
    CT = np.zeros((3*S,3*Ntot),float)
    for i in range(Ntot):
        s = swimmer_id[i]
        wi = w[i]
        CF[3*s:3*s+3, 3*i:3*i+3] += wi * np.eye(3)
        r = points[i] - x0s[s]
        CT[3*s:3*s+3, 3*i:3*i+3] += wi * crossMat(r)
    return CF, CT

def solve_swimmer(points,u_shape,eps,mu=1.0,swimmer_id = None, x0s = None, w=None):
    points = np.asarray(points,float)
    u_shape = np.asarray(u_shape,float)
    if points.ndim !=2 or points.shape[1] != 3:
        raise ValueError("points should be (N,3)")
    if u_shape.shape != points.shape:
        raise ValueError("u_shape must have same shape as points")
    Ntot = points.shape[0]
    if swimmer_id is None:
        swimmer_id = np.zeros(Ntot,dtype = int)
        S = 1
    else:
        swimmer_id = np.asarray(swimmer_id,dtype = int)
        if swimmer_id.shape != (Ntot,):
            raise ValueError("swimmer_id should be (N,)")
        if swimmer_id.min() < 0:
            raise ValueError("swimmer_id should be non-negative integers")
        S = int(swimmer_id.max()) + 1
    x0s = np.asarray(x0s,float)
    if x0s.shape != (S,3):
        raise ValueError("x0s should be (S,3)")
    M = M_mat(points,eps,mu, w=w)
    R_U = buildR_U(swimmer_id,Ntot,S)
    R_Omega = buildR_Omega(points,swimmer_id,x0s,Ntot,S)
    CF, CT = build_constraint_mat(points,swimmer_id,x0s,Ntot,S,w=w)
    ZeroMats = np.zeros((3*S,3*S),float)
    A = np.block([
        [M, -R_U, -R_Omega],
        [CF, ZeroMats, ZeroMats],
        [CT, ZeroMats, ZeroMats]
    ])
    b = np.concatenate([stack_velocities(u_shape),np.zeros(3*S), np.zeros(3*S)])
    sol = np.linalg.solve(A, b)
    
    F = sol[:3*Ntot].reshape(Ntot,3)
    U = sol[3*Ntot:3*Ntot+3*S].reshape(S,3)
    Omega = sol[3*Ntot+3*S:3*Ntot+6*S].reshape(S,3)
    
    return F, U, Omega

def arclength_trapz_weights(points): #arclength-based quadrature weights for integration along curve defined by points
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    w = np.zeros(points.shape[0])
    w[0]  = ds[0]/2
    w[-1] = ds[-1]/2
    w[1:-1] = (ds[:-1] + ds[1:]) / 2
    return w

def weighted_mean(points, w):
    w = np.asarray(w, float)
    return (w[:, None] * points).sum(axis=0) / w.sum()


def fibonacci_sphere_points(N, radius):
    # uniform points on a sphere
    i = np.arange(N)
    golden = (1 + 5**0.5) / 2
    theta = 2*np.pi * i / golden
    z = 1 - 2*(i + 0.5)/N
    r = np.sqrt(1 - z*z)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    pts = np.vstack([x, y, z]).T
    return radius * pts

def sphere_area_weights(N, radius):
    # equal-area quadrature weights for points on a sphere
    return (4*np.pi*radius**2 / N) * np.ones(N)

def prolate_spheroid(N,a,b):
    points_sphere = fibonacci_sphere_points(N, radius=1.0)
    pts = points_sphere * np.array([b,a,a])  # scale x by a and y,z by b
    return pts

def spheroid_weights(sphere_pts,a,b):
    x_hat, y_hat, z_hat = sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2]
    theta = np.arccos(np.clip(z_hat, -1, 1))  # polar angle
    phi = np.arctan2(y_hat, x_hat)  # azimuthal angle

    J = a*np.sqrt(b**2+(a**2-b**2)*np.sin(theta)**2*np.cos(phi)**2)  # Jacobian
    e = np.sqrt(1 - (a/b)**2)  # eccentricity
    S_spheroid = 2*np.pi*a**2+2*np.pi*a*b/e*np.arcsin(e)  # surface area of prolate spheroid

    w = (J/J.sum()) *S_spheroid
    return w


@dataclass
class SphereHead(Component):
    radius: float
    N:int
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))

    pts: np.ndarray = field(init=False,repr=False)
    w: np.ndarray = field(init=False,repr=False)

    def __post_init__(self):
        self.center = np.asarray(self.center,float).reshape(3,)
        self.pts = fibonacci_sphere_points(self.N, self.radius) + self.center
        self.w = sphere_area_weights(self.N, self.radius)
    def body_points(self, t:float) -> np.ndarray:
        return self.pts
    def body_u_shape(self, t:float) -> np.ndarray:
        return np.zeros_like(self.pts)
    def weights(self, t:float) -> np.ndarray:
        return self.w
    
@dataclass
class ProlateSpheroidHead(Component):
    a: float # minor semi-axis (y and z)
    b: float # major semi-axis (x)
    N: int
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))

    pts: np.ndarray = field(init=False,repr=False)
    w: np.ndarray = field(init=False,repr=False)

    def __post_init__(self):
        self.center = np.asarray(self.center,float).reshape(3,)
        sphere_pts = fibonacci_sphere_points(self.N, radius=1.0)
        self.pts = sphere_pts * np.array([self.b,self.a,self.a]) + self.center  # scale x by a and y,z by b
        self.w = spheroid_weights(sphere_pts, self.a, self.b)
    def body_points(self, t:float) -> np.ndarray:
        return self.pts
    
    def body_u_shape(self, t:float) -> np.ndarray:
        return np.zeros_like(self.pts)
    
    def weights(self, t:float) -> np.ndarray:
        return self.w
@dataclass
class Filament(Component):
    stroke: object  # should have r(s,t) and r_t(s,t)
    sgrid: np.ndarray  # (Ns,)
    attatch_offset: np.ndarray
    attatch_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    remove_mean_u:bool = False

    def __post_init__(self):
        self.sgrid = np.asarray(self.sgrid,float)
        self.attatch_offset = np.asarray(self.attatch_offset,float).reshape(3,)
        self.attatch_R = np.asarray(self.attatch_R,float).reshape(3,3)

    def body_points(self, t:float) -> np.ndarray:
        r = self.stroke.r(self.sgrid, t) # (Ns,3)
        return (r@self.attatch_R.T) + self.attatch_offset

    def body_u_shape(self, t:float) -> np.ndarray:
        u = self.stroke.r_t(self.sgrid, t) # (Ns,3)
        u = u @ self.attatch_R.T
        if self.remove_mean_u:
            pts = self.body_points(t)
            w = arclength_trapz_weights(pts)
            u -= weighted_mean(u, w)
        return u

    def weights(self, t:float) -> np.ndarray:
        return arclength_trapz_weights(self.body_points(t))

def run_simulation(swimmers, Xs, Qs, ts, eps, mu, N_head, save_frames=True):
    S = len(swimmers)
    dt = ts[1] - ts[0]
    all_Us = [[] for _ in range(S)]
    all_Omegas = [[] for _ in range(S)]
    all_frames = [[] for _ in range(S)] if save_frames else None
    P_totals = []
    P_usefuls = []

    for t in ts:
        pts_blocks, ush_blocks, w_blocks, id_blocks = [], [], [], []
        for s_idx, sw in enumerate(swimmers):
            r_body, u_body, w = sw.body_state(t)
            pts = Xs[s_idx] + (r_body @ Qs[s_idx].T)
            ush = u_body @ Qs[s_idx].T
            pts_blocks.append(pts)
            ush_blocks.append(ush)
            w_blocks.append(w)
            id_blocks.append(np.full(pts.shape[0], s_idx, dtype=int))

        pts_all = np.vstack(pts_blocks)
        ush_all = np.vstack(ush_blocks)
        w_all = np.concatenate(w_blocks)
        swimmer_id = np.concatenate(id_blocks)
        x0s = np.vstack(Xs)

        F_all, U_all, Omega_all = solve_swimmer(
            pts_all, ush_all, eps=eps, mu=mu,
            swimmer_id=swimmer_id, x0s=x0s, w=w_all
        )

        P_totals.append(np.sum(F_all[N_head:] * ush_all[N_head:] * w_all[N_head:, None]))
        F_drag = (F_all[:N_head] * w_all[:N_head, None]).sum(axis=0)
        P_usefuls.append(np.dot(F_drag, U_all[0]))

        for s_idx in range(S):
            Xs[s_idx] = Xs[s_idx] + U_all[s_idx] * dt
            Qs[s_idx] = rot_from_omega(Omega_all[s_idx], dt) @ Qs[s_idx]
            all_Us[s_idx].append(U_all[s_idx])
            all_Omegas[s_idx].append(Omega_all[s_idx])
            if save_frames:
                all_frames[s_idx].append(pts_blocks[s_idx])

    all_Us = [np.array(u) for u in all_Us]
    all_Omegas = [np.array(om) for om in all_Omegas]
    if save_frames:
        all_frames = [np.array(fr) for fr in all_frames]

    mean_P = np.mean(P_totals)
    mean_eta = np.mean(P_usefuls) / mean_P
    return all_Us, all_Omegas, all_frames, mean_P, mean_eta

