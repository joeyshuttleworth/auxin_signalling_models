function ReducedAuxinSignallingPathway_f_deriv!(dz, z, p, t=0.0)
	(ARF_1, D__ARF_1__ARF_1, D__ARF_1__iaa_1, D__iaa_1__iaa_1, R__iaa_1, iaa_1) = z

	(d__D__ARF_1__ARF_1, d__D__ARF_1__iaa_1, d__D__iaa_1__iaa_1, d__G__ARF_1__ARF_1, d__G__ARF_1__iaa_1, d__R, d__arf__all, d__iaa_1__x, d__iaa__all, k__D__ARF_1__ARF_1, k__D__ARF_1__iaa_1, k__D__iaa_1__iaa_1, k__G__ARF_1__ARF_1, k__G__ARF_1__ARF_1__R__iaa_1, k__G__ARF_1__R__iaa_1, k__G__ARF_1__iaa_1, k__G__ARF_1__iaa_1__R__iaa_1, k__G__R__iaa_1, k__RNA__ARF, k__RNA__IAA, k__arf__all, x) = p

	x0 = D__ARF_1__ARF_1 .* k__G__ARF_1__ARF_1
	x1 = 1 ./ (-d__G__ARF_1__ARF_1 - x0)
	x2 = D__ARF_1__iaa_1 .* k__G__ARF_1__iaa_1
	x3 = x0 .* x1 .* x2 + x2
	x4 = x3 ./ (-d__G__ARF_1__iaa_1 - x3)
	(G__ARF_1__ARF_1, G__ARF_1__iaa_1) = [-x1 .* (x0 .* x4 + x0), -x4]

	 G = 1 - (G__ARF_1__ARF_1 + G__ARF_1__iaa_1)

	dz .= (-2 * ARF_1 .^ 2 .* k__D__ARF_1__ARF_1 - ARF_1 .* d__arf__all - ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 + 2 * D__ARF_1__ARF_1 .* d__D__ARF_1__ARF_1 + D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1 + k__arf__all, ARF_1 .^ 2 .* k__D__ARF_1__ARF_1 - D__ARF_1__ARF_1 .* d__D__ARF_1__ARF_1, ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 - D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1, -D__iaa_1__iaa_1 .* d__D__iaa_1__iaa_1 + iaa_1 .^ 2 .* k__D__iaa_1__iaa_1, G .* k__G__R__iaa_1 + G__ARF_1__ARF_1 .* k__G__ARF_1__ARF_1__R__iaa_1 + G__ARF_1__iaa_1 .* k__G__ARF_1__iaa_1__R__iaa_1 - R__iaa_1 .* d__R, -ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 + D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1 + 2 * D__iaa_1__iaa_1 .* d__D__iaa_1__iaa_1 + R__iaa_1 .* k__RNA__IAA - d__iaa_1__x .* iaa_1 .* x - d__iaa__all .* iaa_1 - 2 * iaa_1 .^ 2 .* k__D__iaa_1__iaa_1)

end