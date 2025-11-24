clear; clc; close all;

s = tf('s');

K = (6136*s + 108700)/(s^3 + 89*s^2 + 9258*s + 108700);

Kss = ss(K)