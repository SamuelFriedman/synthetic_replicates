#!/usr/bin/python3
############################################################
#
# Originally created by Samuel Friedman, March 1, 2016
# while at the Lawrence J. Ellison Institute for Transformative
# Medicine of USC (University of Southern California).
#
# Copyright 2017, Samuel Friedman, Opto-Knowledge Systems, Inc. (OKSI)
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
############################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import optimize
from scipy import stats

initial_ratio = 1e-5
growth_rate = 1.0/30.0
growth_time = 1.0/growth_rate

n0_variability_percentage = 0.05
count_percentage_uncertainty = 0.05
time_uncertainty = 1.0
growth_rate_variability_percentage = 0.05

num_technical_replicates = 20
num_biological_replicates = 5
num_days = 3
day_values = np.array([1.0, 2.0, 3.0])
num_total_replicates = num_biological_replicates*num_technical_replicates
num_values = num_days*num_total_replicates
num_hours_in_day = 24.0

ratios = [initial_ratio, initial_ratio*1e1, initial_ratio*1e2, initial_ratio*1e3, initial_ratio*1e4, initial_ratio*5e4]
#ratios = [initial_ratio*5e4]
#ratios = [initial_ratio, initial_ratio*1e1]
num_ratios = len(ratios)

np.random.seed(27)

# Requires x data points, y data points, and the base standard deviations for x and y
# Returns (slope, y_intercept)
def fit_deming_errors(x, y, delta_x, delta_y):
    cov = np.cov(x, y)
    delta = np.square(delta_y / delta_x)
    gamma = (cov[1,1] - delta*cov[0,0])/(2.0*cov[1,0])
    slope = gamma + cov[1,0]*np.sqrt(delta + gamma*gamma)
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    y_int = y_bar - slope*x_bar
    return (slope, y_int)

# Fit x and y with weights w_x and w_y. Note, w_x and w_y (in this implementation) need to be uncorrelated
# This algorithm is from York et al., 2004
def fit_york_errors(x, y, w_x, w_y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    b = optimize.newton(york_b_diff, slope, args=(w_x, w_y, x, y),tol=1e-15)
    #print(york_b_diff(b, w_x, w_y, x, y))
    w_i = w_x*w_y/(w_x+b*b*w_y)
    X_bar = np.average(x, weights=w_i)
    Y_bar = np.average(y, weights=w_i)
    y_int = Y_bar - b*X_bar
    U = x - X_bar
    V = y - Y_bar
    beta_i = w_i*(U/w_y+b*V/w_x)
    x_i = X_bar + beta_i
    x_bar_small = np.average(x_i, weights=w_i)
    u_i = x_i - x_bar_small
    sigma_b = 1.0/np.sum(w_i*np.square(u_i))
    return (b, y_int, sigma_b)

# This function computes the rearranged form of Eqn 13b so that we deal with root finding instead of value replacement
def york_b_diff(b, w_x, w_y, X, Y):
    w_i = w_x*w_y/(w_x+b*b*w_y)
    X_bar = np.average(X, weights=w_i)
    Y_bar = np.average(Y, weights=w_i)
    U = X - X_bar
    V = Y - Y_bar
    beta_i = w_i*(U/w_y+b*V/w_x)
    return np.sum(w_i*beta_i*V)/np.sum(w_i*beta_i*U) - b

#def main():

overall_slopes = np.empty(num_ratios)
overall_slope_errors = np.empty(num_ratios)
overall_y_ints = np.empty(num_ratios)
overall_linregress = np.empty(num_ratios)
overall_linregress_yint = np.empty(num_ratios)
ratio_index = 0
time_values = np.empty((num_ratios, num_values))
count_values = np.empty((num_ratios, num_values))
replicate_slopes = np.empty((num_ratios, num_total_replicates))
replicate_slope_errors = np.empty((num_ratios, num_total_replicates))
replicate_linregress = np.empty((num_ratios, num_total_replicates))
max_time = 120
times = np.linspace(0,max_time)
times_length = np.shape(times)[0]
growth_curves = np.empty((num_ratios, times_length))

# For each rough ratio
for ratio in ratios:
    value_index = 0
    replicate_index = 0

    # For each biological growth_rate
    for i in range(num_biological_replicates):
        current_growth_rate = 1.0/np.random.normal(growth_time, growth_time*growth_rate_variability_percentage)
        # For each technical replicate
        for j in range(num_technical_replicates):
            n0_inv_ratio = 1.0/np.random.normal(ratio, n0_variability_percentage*ratio)
            #current_counts = np.log(n0_inv_ratio/(1.0 - (1.0 - n0_inv_ratio)*np.exp(-growth_rate*day_values*num_hours_in_day)))
            current_counts = (1.0/(1.0 - (1.0 - n0_inv_ratio)*np.exp(-growth_rate*day_values*num_hours_in_day)))
            #current_counts = n0_inv_ratio/(1.0 - (1.0 - n0_inv_ratio)*np.exp(-growth_rate*day_values*num_hours_in_day)) # Renormalized by n0_inv_ratio

            replicate_times = np.empty(num_days)
            replicate_counts = np.empty(num_days)

            # For each day
            for day in range(num_days):
                time = np.random.normal(day_values[day]*num_hours_in_day, time_uncertainty)
                time_values[ratio_index, value_index] = time
                replicate_times[day] = time
                count = np.log(np.random.normal(current_counts[day], current_counts[day]*count_percentage_uncertainty))
                #count = (np.random.normal(current_counts[day], current_counts[day]*count_percentage_uncertainty))
                count_values[ratio_index, value_index] = count
                replicate_counts[day] = count

                #current_index += 1
                value_index += 1

            # Do a fit
            w_x = 1.0/np.square(np.full_like(replicate_times, time_uncertainty))
            w_y = 1.0/np.square(replicate_counts*count_percentage_uncertainty)
            b, y_int, sigma_b = fit_york_errors(replicate_times, replicate_counts, w_x, w_y)
            replicate_slopes[ratio_index, replicate_index] = b
            replicate_slope_errors[ratio_index, replicate_index] = sigma_b
            slope, intercept, r_value, p_value, std_err = stats.linregress(replicate_times, replicate_counts)
            replicate_linregress[ratio_index, replicate_index] = slope

            replicate_index += 1

    # Now fit over all of the replicates
    w_x = 1.0/np.square(np.full_like(time_values[ratio_index,:], time_uncertainty))
    w_y = 1.0/np.square(count_values[ratio_index,:]*count_percentage_uncertainty)
    b, y_int, sigma_b = fit_york_errors(time_values[ratio_index,:], count_values[ratio_index,:], w_x, w_y)
    overall_slopes[ratio_index] = b
    overall_slope_errors[ratio_index] = sigma_b
    overall_y_ints[ratio_index] = y_int
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_values[ratio_index,:], count_values[ratio_index,:])
    overall_linregress[ratio_index] = slope
    overall_linregress_yint[ratio_index] = intercept
    growth_curves[ratio_index,:] = np.log(1.0/(1.0-(1.0-n0_inv_ratio)*np.exp(-growth_rate*times)))

    ratio_index += 1

############################################################
# PLOTS
############################################################

##############################
# Plot 1
##############################

color_letters = ('r', 'y', 'g', 'c', (0.5,0.5,1.0), 'm')
std_dev_x = np.zeros((num_ratios, num_days))
std_dev_y = np.zeros((num_ratios, num_days))
cloud_x = np.zeros((num_ratios, num_days))
cloud_y = np.zeros((num_ratios, num_days))
plt.figure(figsize=(6,8))
plot_left = 0.15
plot_right = 0.95
plt.subplots_adjust(left=plot_left, right=plot_right)
for i in range(num_ratios):
    for j in range(num_days):
        std_dev_x[i,j] = np.std(time_values[i,j:-1:3])
        std_dev_y[i,j] = np.std(np.exp(count_values[i,j:-1:3]))
        cloud_x[i,j] = np.mean(time_values[i,j:-1:3])
        cloud_y[i,j] = np.mean(np.exp(count_values[i,j:-1:3]))
        #    plt.scatter(time_values[i,:], count_values[i,:], s=5, c=colors[i,:], alpha=0.5)
        #    plt.plot(times, growth_curves[i,:], c=colors[i,:])
    #plt.scatter(time_values[i,:], np.exp(count_values[i,:]), s=20, c=color_letters[i], edgecolor='face', alpha=0.1, zorder=1)
    plt.scatter(time_values[i,:], np.exp(count_values[i,:]), s=10, c=color_letters[i], alpha=0.6, edgecolor='black', zorder=1)
    #plt.plot(times, np.exp(growth_curves[i,:]),'--', c=color_letters[i], zorder=4)
    num_zeros = (5-i)
    plt.plot(times, np.exp(overall_linregress[i]*times+overall_linregress_yint[i]), c=color_letters[i], zorder=2, label=("{:."+str((5-i)//2*2+1)+"f}").format(ratios[i]) ) #label=r"$N_0/N_{{cap}}$ = {:1.0e}".format(ratios[i]) # FOR PYTHON 3
    #plt.plot(times, np.exp(overall_linregress[i]*times+overall_linregress_yint[i]), c=color_letters[i], zorder=2, label=("{:."+str((5-i)/2*2+1)+"f}").format(ratios[i]) ) #label=r"$N_0/N_{{cap}}$ = {:1.0e}".format(ratios[i]) # FOR PYTHON 2
    plt.errorbar(cloud_x[i,:], cloud_y[i,:], xerr=std_dev_x[i,:], ms=5, yerr=std_dev_y[i,:], c='k', marker='', ls='None', zorder=3, elinewidth=6.0)
    plt.errorbar(cloud_x[i,:], cloud_y[i,:], xerr=std_dev_x[i,:], ms=5, yerr=std_dev_y[i,:], c='w', marker='', ls='None', zorder=3, elinewidth=2.0)

plt.legend(loc='lower right', ncol=3, fancybox=False,numpoints=2,fontsize='large',framealpha=None,edgecolor='inherit')
plt.semilogy()
plt.xlim(15,80)
plt.ylim(3e-6,2)
plt.ylabel(r'Cell Counts/$N_{cap}$')
plt.xlabel('Time [Hours]')

plt.savefig('plot1.png')

##############################
# Plot 2
##############################

avg_slopes = np.average(replicate_slopes,1)
avg_linregress = np.average(replicate_linregress,1)
std_slopes = np.std(replicate_slopes,1)
std_linregress = np.std(replicate_linregress,1)
num_y_figs = 4
plt.figure(figsize=(6,8))
ax1 = plt.subplot2grid((num_y_figs,1),(0,0),rowspan=num_y_figs-1)
plt.errorbar(ratios, avg_slopes, yerr=std_slopes, marker='o', label='Fitting using errors', color='b', capsize=3, markeredgewidth=1.0)
plt.errorbar(ratios, avg_linregress, yerr=std_linregress, marker='o', label='Fitting without errors', color='g', capsize=3, markeredgewidth=1.0)
plt.legend(loc='lower left',fancybox=False,numpoints=2,fontsize='large',framealpha=None,edgecolor='inherit')
#plt.plot(ratios, overall_slopes, '-o')
#plt.plot(ratios, overall_linregress, '-o')
plt.xscale('log')
plt.ylabel(r'Proliferation Rate [hour$^{-1}$]')
plt.axhline(growth_rate,color='r') #linestyle='dashed'
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
xmin = 4e-6
xmax = 2.0
plt.xlim((xmin, xmax))
ymin, ymax = (0.00275, 0.0375)
plt.ylim((ymin,ymax))
hist_xmin = 0.0
hist_xmax = 0.04
hist_ymin = 0.0
hist_ymax = 30
hist_size = 0.075
plot_left = 0.15
plot_right = 0.97
plt.subplots_adjust(left=plot_left, right=plot_right)
subplot_font_size = 'small'
for plt_index in range(num_ratios):
    plot_x_frac = (np.log(ratios[plt_index])-np.log(xmin))/(np.log(xmax)-np.log(xmin))
    plot_x_pos = (plot_right-plot_left)*plot_x_frac + plot_left - 0.005
    plot_y_pos = 0.55
    if(plt_index == 4):
        plot_y_pos = 0.735
    axes_pos = [plot_x_pos, plot_y_pos, hist_size, hist_size]
    a = plt.axes(axes_pos)
    plt.hist(replicate_slopes[plt_index,:],color='b')
    plt.axvline(growth_rate,color='r')
    plt.xlim((hist_xmin, hist_xmax))
    plt.ylim((hist_ymin, hist_ymax))
    plt.xticks([0.0, 0.02, 0.04], rotation='vertical', size=subplot_font_size) #[0.0, 0.01, 0.02, 0.03, 0.04]
    plt.yticks([0, 20], size=subplot_font_size)
    if(plt_index == 4):
        plt.xlabel("Prolif Rate", size=subplot_font_size)
        plt.ylabel("Num Obs", size=subplot_font_size)
        #a.axhspan(0.30, 0.40, xmin=plot_x_pos, xmax=plot_x_pos+hist_size, fc='blue', zorder=2000)
plt.subplot2grid((num_y_figs,1),(num_y_figs-1,0))
plt.xlabel(r'$N_0/N_{cap}$')
plt.xlim((xmin, xmax))
plt.xscale('log')
plt.subplots_adjust(hspace=0.0)
p_values = np.empty(num_ratios-1)
for i in range(num_ratios-1):
    p_values[i] = stats.ttest_ind(replicate_slopes[0,:], replicate_slopes[i+1,:])[1]
plt.scatter(ratios[1:], p_values, marker='x', s=200, linewidth=3.0, label="Student's t-test", color='b')
plt.ylabel(r'$p$-value')
plt.ylim((0.0, 0.5))
plt.axhline(0.05,linestyle='dashed',color='b')
plt.legend(loc="upper right", numpoints=1, scatterpoints=1, scatteryoffsets=[0.5],fancybox=False,fontsize='large',framealpha=None,edgecolor='inherit')

plt.savefig('plot2.png')
