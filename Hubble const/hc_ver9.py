import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt

plt.rcParams['figure.figsize'] = [10, 15]
fig, (ax1, ax2) = plt.subplots(2, 1)

#%%##############################Step 1#####################################
#--------------------------------Step 1-------------------------------------
# Get data
def load_data(file_path):
    """Load data from the specified file and return relevant columns."""
    star_name = np.loadtxt(file_path, 
                           unpack=True, 
                           usecols=(0), 
                           skiprows=3, 
                           dtype=str
                           )
    par, par_err, period, vis_mag, ext, ext_err = np.loadtxt(file_path, 
                                                             unpack=True, 
                                                             usecols=(1, 2, 3, 4, 5, 6), 
                                                             skiprows=3, 
                                                             dtype=float
                                                             )
    
    return star_name, par, par_err, period, vis_mag, ext, ext_err

star_name, par, par_err, period, vis_mag, ext, ext_err = load_data('MW_Cepheids.dat')

# Chi^2
def chisq(y, y_m, y_err, num_par):
    """Calculate the chi-squared and reduced chi-squared values.

    Args:
        y (array): Observed values.
        y_m (array): Model values.
        y_err (array): Errors associated with the observed values.
        num_par (int): Number of parameters in the model.

    Returns:
        tuple: chi-squared and reduced chi-squared values.
    """
    chi2 = np.sum(((y - y_m) / y_err) ** 2)
    rchi2 = chi2 / (len(y) - num_par)
    return chi2, rchi2

# Uncertainty
def uncer(name, list_a, list_b, eq, var_name, unit, list_c=None, list_d=None, *args):
    """Calculate uncertainties in derived quantities using Monte Carlo simulations.

    Args:
        name (list): Names of the objects.
        list_a (array): Array of input values.
        list_b (array): Array of associated errors for input values.
        eq (function): Function to compute the derived quantity.
        var_name (str): Name of the derived variable.
        unit (str): Unit of the derived variable.
        list_c (array, optional): Second input array for the equation.
        list_d (array, optional): Associated errors for the second input array.
        *args: Additional arguments for the equation.

    Returns:
        tuple: Mean and standard deviation of the derived quantities.
    """
    result, result_err = [], []
    
    for i in range(len(list_a)):
        sim_1 = np.random.normal(list_a[i], list_b[i], size=100000)
        
        if list_c is not None and list_d is not None:
            sim_2 = np.random.normal(list_c[i], list_d[i], size=100000)
        
        if len(args) == 0:
            if list_c is not None and list_d is not None:
                x = eq(sim_1, sim_2)
            else:
                x = eq(sim_1)
        else:
            cur_args = [arg[i] for arg in args]
            if list_c is not None and list_d is not None:
                x = eq(sim_1, sim_2, *cur_args)
            else:
                x = eq(sim_1, *cur_args)
        
        ele_m, ele_e = np.mean(x), np.std(x)
        
        print(name[i], ":", var_name, "=", round(ele_m, 2), "±", round(ele_e, 2), unit)
        result.append(ele_m)
        result_err.append(ele_e)
    
    return result, result_err

# Cal curve info
def get_curve_info(eq, x, y, yerr):
    """Fit a curve to the data and return the parameters and best-fit line.

    Args:
        eq (function): The equation to fit.
        x (array): Independent variable data.
        y (array): Dependent variable data.

    Returns:
        tuple: Slope, intercept, their errors, and the best-fit line.
    """
    param, cov = opt.curve_fit(eq, x, y, sigma=yerr)
    
    if len(param) == 1:
        grad, y_int = param, 0
        grad_err, y_int_err = np.sqrt(cov), 0
        best_fit_line = eq(x, grad)
    else:
        grad, y_int = param
        grad_err, y_int_err = np.sqrt(np.diag(cov))
        best_fit_line = eq(x, grad, y_int)
    
    return grad, y_int, grad_err, y_int_err, best_fit_line

# Calculate distance
def cal_d(p):
    """Calculate distance in parsecs from the period.

    Args:
        p (float): Parallax in milli-arcsecs.

    Returns:
        float: Distance in parsecs.
    """
    return 1000 / p

# Calculate absolute magnitude
def cal_abs_mag(d, A, m):
    """Calculate the absolute magnitude from distance, extinction, and apparent magnitude.

    Args:
        d (float): Distance in parsecs.
        A (float): Extinction.
        m (float): Apparent magnitude.

    Returns:
        float: Absolute magnitude.
    """
    return m - 5 * np.log10(d) + 5 - A

# Distance Calculation
print("--------Distance from Earth--------")
d_pc, d_pc_err = uncer(star_name, 
                       par, 
                       par_err, 
                       cal_d, 
                       "d_pc", 
                       "pc"
                       )

# Extinction Information
print("\n-------------Extinction------------")
for i in range(10):
    print(star_name[i], r": A =", round(ext[i], 2), "±", round(ext_err[i], 2), "mag")
    
# Absolute Magnitude Calculation
print("\n--------Absolute Magnitude---------")
abs_mag, abs_mag_err = uncer(star_name,
                            d_pc, 
                            d_pc_err,
                            cal_abs_mag, 
                            "M",
                            "mag",
                            ext,
                            ext_err, 
                            vis_mag
                            )

def lin_func(x, m, c):
    """Linear function for fitting.

    Args:
        x (array): Independent variable.
        m (float): Slope of the line.
        c (float): Intercept of the line.

    Returns:
        array: Values of the linear function.
    """
    return m * x + c

# Fit the period-luminosity relation
log_period = np.log10(period)
grad, y_int, grad_err, y_int_err, best_fit_line = get_curve_info(lin_func, 
                                                                log_period, 
                                                                abs_mag,
                                                                abs_mag_err
                                                                )

# Calculate chi-squared values
chi_sq, rchi_sq = chisq(abs_mag, best_fit_line, abs_mag_err, 2)
print(chi_sq, rchi_sq)

print("\n-----Period-Luminosity Relation----")
print(f"M = {grad} * log10(P) + {y_int}")
print(f"α = {round(grad, 2)} ± {round(grad_err, 2)}", 
      f"\nβ = {round(y_int, 2)} ± {round(y_int_err, 2)}")

# Plot the period-luminosity relationship
ax1.set_title("Cepheid period-luminosity relationships")
ax1.scatter(log_period, abs_mag, label="Data points")
ax1.errorbar(log_period, abs_mag, yerr=abs_mag_err, linestyle="None")
ax1.plot(log_period, best_fit_line, color="red", label="Best fit line")
ax1.set_xlabel("Log10(P) / days")
ax1.set_ylabel("M / mag")
str1 = r'$\chi^2 =$' + str(np.round(chi_sq, 2))
ax1.text(1.2, -3.3, str1, fontsize=20)
str2 = r'Reduced $\chi^2 =$' + str(np.round(rchi_sq, 2))
ax1.text(1.2, -3, str2, fontsize=20)
ax1.invert_yaxis()
ax1.legend()
#plt.show()

#%%##############################Step 2#####################################
#--------------------------------Step 2-------------------------------------
def load_ceph_data(file_path):
    """Load Cepheid data from the specified file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        tuple: Names, log10 periods, and apparent magnitudes of Cepheids.
    """
    ceph_name = np.loadtxt(file_path, unpack=True, usecols=(0), skiprows=1, dtype=str)
    log10_P_ceph, app_mag_ceph = np.loadtxt(file_path, unpack=True, usecols=(1, 2), skiprows=1, dtype=float)
    return ceph_name, log10_P_ceph, app_mag_ceph

ceph_name, log10_P_ceph, app_mag_ceph = load_ceph_data('ngc4527_cepheids.dat')

# Uncertainty calculation for Cepheids
def uncer_ceph(name, g, g_err, y, y_err, eq, *args):
    """Calculate uncertainties for Cepheid distances using Monte Carlo simulations.

    Args:
        name (list): Names of the Cepheids.
        g (float): Slope of the period-luminosity relation.
        g_err (float): Error in the slope.
        y (float): Intercept of the period-luminosity relation.
        y_err (float): Error in the intercept.
        eq (function): Function to calculate the distance.
        *args: Additional arguments for the distance equation.

    Returns:
        tuple: Mean and standard deviation of the calculated distances.
    """
    result, result_err = [], []
    
    for i in range(len(name)):
        # Simulate the slope and intercept based on their uncertainties
        sim_g = np.random.normal(g, g_err, size=100000)
        sim_y = np.random.normal(y, y_err, size=100000)
        
        # Calculate distance using the simulated values
        x = cal_ceph_d(sim_g, sim_y, app_mag_ceph[i], A_mw, log10_P_ceph[i])
        
        ele_m, ele_e = np.mean(x) / 10**6, np.std(x) / 10**6  # Convert to Mpc
        
        print(name[i], ": d_pc =", round(ele_m, 2), "±", round(ele_e, 2), "Mpc")
        result.append(ele_m)
        result_err.append(ele_e)
        
    return result, result_err

def cal_ceph_d(a, b, m, A, log10_P):
    """Calculate distance to Cepheids using the period-luminosity relation.

    Args:
        a (float): Slope of the relation.
        b (float): Intercept of the relation.
        m (float): Apparent magnitude of the Cepheid.
        A (float): Extinction.
        log10_P (float): Logarithm of the period.

    Returns:
        float: Distance to the Cepheid in parsecs.
    """
    x = 10**((m + 5 - A - a * log10_P - b) / 5)
    return x

A_mw = 0.0682  # Milky Way extinction

print("\n----------NGC4527 Cepheid----------")
d_pc_ceph, d_pc_ceph_err = uncer_ceph(ceph_name,
                                       grad, 
                                       grad_err, 
                                       y_int, 
                                       y_int_err,
                                       app_mag_ceph, 
                                       A_mw,
                                       log10_P_ceph,
                                       )

d_pc_ceph_m = np.mean(d_pc_ceph)
d_pc_ceph_err_m = np.mean(d_pc_ceph_err)

#%%##############################Step 3#####################################
#--------------------------------Step 3-------------------------------------
def load_galaxy_data(file_path):
    """Load galaxy recession velocity and distance data from the specified file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        tuple: Galaxy names, recession velocities, distances, and errors.
    """
    gal_name = np.loadtxt(file_path, unpack=True, usecols=(0), skiprows=2, dtype=str)
    v_rec = np.loadtxt(file_path, unpack=True, usecols=(1), skiprows=2, dtype=int)
    d_Mpc_gal, d_Mpc_gal_err = np.loadtxt(file_path, unpack=True, usecols=(2, 3), skiprows=2, dtype=float)
    return gal_name, v_rec, d_Mpc_gal, d_Mpc_gal_err

gal_name, v_rec, d_Mpc_gal, d_Mpc_gal_err = load_galaxy_data('other_galaxies.dat')

def cal_H0(D_gal, v_rec):
    """Calculate the Hubble constant from distance and recession velocity.

    Args:
        D_gal (float): Distance to the galaxy in Mpc.
        v_rec (float): Recession velocity of the galaxy in km/s.

    Returns:
        float: Hubble constant in km/s/Mpc.
    """
    return v_rec / D_gal

def lin_func_H0(x, m):
    """Linear function for Hubble's law.

    Args:
        x (array): Independent variable.
        m (float): Slope of the line.

    Returns:
        array: Dependent variable (e.g., velocity).
    """
    return m * x

v_rec_NGC4527 = 1152  # Recession velocity for NGC4527

# Append NGC4527 data to existing arrays
new_gal_name = np.append(gal_name, "HGC4527")
new_v_rec = np.append(v_rec, v_rec_NGC4527)
new_d_Mpc = np.append(d_Mpc_gal, d_pc_ceph_m)
new_d_Mpc_err = np.append(d_Mpc_gal_err, d_pc_ceph_err_m)

# Print recession velocities
print("\n-------------Galaxies--------------")
print("--------Recession Velocity---------")
for i in range(len(new_gal_name)):
    print(f"{new_gal_name[i]} : v_rec = {round(np.mean(new_v_rec[i]), 2)} km/s")

# Print distances
print("\n-------------Distance--------------")
for i in range(len(new_gal_name)):
    print(f"{new_gal_name[i]} : d_Mpc = {round(np.mean(new_d_Mpc[i]), 2)} ± {round(new_d_Mpc_err[i], 2)} Mpc")

# Hubble Constant Calculation
print("\n----------Hubble Constant----------")
H0, H0_err = uncer(new_gal_name,
                   new_d_Mpc,
                   new_d_Mpc_err,
                   cal_H0,
                   "H0",
                   "km/s/Mpc",
                   None,
                   None,
                   new_v_rec
                   )

print(f"\nH0 = {round(np.mean(H0), 2)} ± {round(np.mean(H0_err), 2)} km/s/Mpc")

# Fit the Hubble diagram
grad, y_int, grad_err, y_int_err, best_fit_line = get_curve_info(lin_func_H0, 
                                                                new_v_rec, 
                                                                new_d_Mpc,
                                                                new_d_Mpc_err
                                                                )
h0_best = 1/grad
print(best_fit_line)

# Calculate chi-squared values for the Hubble diagram
chi_sq, rchi_sq = chisq(new_d_Mpc, new_v_rec/h0_best, new_d_Mpc_err, 2)
print(chi_sq, rchi_sq)

# Plot the Hubble diagram
ax2.set_title("Hubble diagram")
ax2.scatter(new_d_Mpc, new_v_rec, label="Data points")
ax2.errorbar(new_d_Mpc, new_v_rec, xerr=new_d_Mpc_err, linestyle="None")
ax2.plot(new_d_Mpc, new_d_Mpc*h0_best, color="red", label="Best fit line")
ax2.set_xlabel("D_gal / 10^6 pc")
ax2.set_ylabel("v_rec / km/s")
str1 = r'$\chi^2 =$' + str(np.round(chi_sq, 2))
ax2.text(15.2, 500, str1, fontsize=20)
str2 = r'Reduced $\chi^2 =$' + str(np.round(rchi_sq, 2))
ax2.text(15.2, 350, str2, fontsize=20)
ax2.legend()
#plt.show()

# Final layout adjustments and display
plt.tight_layout()
plt.show()