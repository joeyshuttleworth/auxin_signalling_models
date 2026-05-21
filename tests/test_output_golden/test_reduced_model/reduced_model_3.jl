function ReducedAuxinSignallingPathway_f_deriv!(dz, z, p, t=0.0)
	(ARF_1, D__ARF_1__ARF_1, D__ARF_1__iaa_1, D__ARF_1__iaa_2, D__iaa_1__iaa_1, D__iaa_1__iaa_2, D__iaa_2__iaa_2, R__ARF_1, R__iaa_1, R__iaa_2, iaa_1, iaa_2) = z

	(d__D__ARF_1__ARF_1, d__D__ARF_1__iaa_1, d__D__ARF_1__iaa_2, d__D__iaa_1__iaa_1, d__D__iaa_1__iaa_2, d__D__iaa_2__iaa_2, d__G__ARF_1__ARF_1, d__G__ARF_1__iaa_1, d__G__ARF_1__iaa_2, d__R, d__arf__all, d__iaa_1__x, d__iaa_2__x, d__iaa__all, k__D__ARF_1__ARF_1, k__D__ARF_1__iaa_1, k__D__ARF_1__iaa_2, k__D__iaa_1__iaa_1, k__D__iaa_1__iaa_2, k__D__iaa_2__iaa_2, k__G__ARF_1__ARF_1, k__G__ARF_1__ARF_1__R__ARF_1, k__G__ARF_1__ARF_1__R__iaa_1, k__G__ARF_1__ARF_1__R__iaa_2, k__G__ARF_1__R__ARF_1, k__G__ARF_1__R__iaa_1, k__G__ARF_1__R__iaa_2, k__G__ARF_1__iaa_1, k__G__ARF_1__iaa_1__R__ARF_1, k__G__ARF_1__iaa_1__R__iaa_1, k__G__ARF_1__iaa_1__R__iaa_2, k__G__ARF_1__iaa_2, k__G__ARF_1__iaa_2__R__ARF_1, k__G__ARF_1__iaa_2__R__iaa_1, k__G__ARF_1__iaa_2__R__iaa_2, k__G__R__ARF_1, k__G__R__iaa_1, k__G__R__iaa_2, k__RNA__ARF, k__RNA__IAA, x) = p

	x0 = D__ARF_1__ARF_1 .* k__G__ARF_1__ARF_1
	x1 = 1 ./ (-d__G__ARF_1__ARF_1 - x0)
	x2 = D__ARF_1__iaa_1 .* k__G__ARF_1__iaa_1
	x3 = x0 .* x1
	x4 = x2 .* x3 + x2
	x5 = 1 ./ (-d__G__ARF_1__iaa_1 - x4)
	x6 = D__ARF_1__iaa_2 .* k__G__ARF_1__iaa_2
	x7 = x3 .* x6 + x6
	x8 = -x5 .* x7
	x9 = -x4
	x10 = (-x4 .* x8 + x7) ./ (-d__G__ARF_1__iaa_2 - x7 - x8 .* x9)
	x11 = x5 .* (-x10 .* x9 + x4)
	(G__ARF_1__ARF_1, G__ARF_1__iaa_1, G__ARF_1__iaa_2) = [-x1 .* (x0 .* x10 + x0 .* x11 + x0), -x11, -x10]

	 G = 1 - (G__ARF_1__ARF_1 + G__ARF_1__iaa_1 + G__ARF_1__iaa_2)

	dz .= (-2 * ARF_1 .^ 2 .* k__D__ARF_1__ARF_1 - ARF_1 .* d__arf__all - ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 - ARF_1 .* iaa_2 .* k__D__ARF_1__iaa_2 + 2 * D__ARF_1__ARF_1 .* d__D__ARF_1__ARF_1 + D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1 + D__ARF_1__iaa_2 .* d__D__ARF_1__iaa_2 + R__ARF_1 .* k__RNA__ARF, ARF_1 .^ 2 .* k__D__ARF_1__ARF_1 - D__ARF_1__ARF_1 .* d__D__ARF_1__ARF_1, ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 - D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1, ARF_1 .* iaa_2 .* k__D__ARF_1__iaa_2 - D__ARF_1__iaa_2 .* d__D__ARF_1__iaa_2, -D__iaa_1__iaa_1 .* d__D__iaa_1__iaa_1 + iaa_1 .^ 2 .* k__D__iaa_1__iaa_1, -D__iaa_1__iaa_2 .* d__D__iaa_1__iaa_2 + iaa_1 .* iaa_2 .* k__D__iaa_1__iaa_2, -D__iaa_2__iaa_2 .* d__D__iaa_2__iaa_2 + iaa_2 .^ 2 .* k__D__iaa_2__iaa_2, G .* k__G__R__ARF_1 + G__ARF_1__ARF_1 .* k__G__ARF_1__ARF_1__R__ARF_1 + G__ARF_1__iaa_1 .* k__G__ARF_1__iaa_1__R__ARF_1 + G__ARF_1__iaa_2 .* k__G__ARF_1__iaa_2__R__ARF_1 - R__ARF_1 .* d__R, G .* k__G__R__iaa_1 + G__ARF_1__ARF_1 .* k__G__ARF_1__ARF_1__R__iaa_1 + G__ARF_1__iaa_1 .* k__G__ARF_1__iaa_1__R__iaa_1 + G__ARF_1__iaa_2 .* k__G__ARF_1__iaa_2__R__iaa_1 - R__iaa_1 .* d__R, G .* k__G__R__iaa_2 + G__ARF_1__ARF_1 .* k__G__ARF_1__ARF_1__R__iaa_2 + G__ARF_1__iaa_1 .* k__G__ARF_1__iaa_1__R__iaa_2 + G__ARF_1__iaa_2 .* k__G__ARF_1__iaa_2__R__iaa_2 - R__iaa_2 .* d__R, -ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 + D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1 + 2 * D__iaa_1__iaa_1 .* d__D__iaa_1__iaa_1 + D__iaa_1__iaa_2 .* d__D__iaa_1__iaa_2 + R__iaa_1 .* k__RNA__IAA - d__iaa_1__x .* iaa_1 .* x - d__iaa__all .* iaa_1 - 2 * iaa_1 .^ 2 .* k__D__iaa_1__iaa_1 - iaa_1 .* iaa_2 .* k__D__iaa_1__iaa_2, -ARF_1 .* iaa_2 .* k__D__ARF_1__iaa_2 + D__ARF_1__iaa_2 .* d__D__ARF_1__iaa_2 + D__iaa_1__iaa_2 .* d__D__iaa_1__iaa_2 + 2 * D__iaa_2__iaa_2 .* d__D__iaa_2__iaa_2 + R__iaa_2 .* k__RNA__IAA - d__iaa_2__x .* iaa_2 .* x - d__iaa__all .* iaa_2 - iaa_1 .* iaa_2 .* k__D__iaa_1__iaa_2 - 2 * iaa_2 .^ 2 .* k__D__iaa_2__iaa_2)

end