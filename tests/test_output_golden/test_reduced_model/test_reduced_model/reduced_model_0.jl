function ReducedAuxinSignallingPathway_f_deriv!(dz, z, p, t=0.0)
	(ARF_1, D__ARF_1__ARF_1, D__ARF_1__iaa_1, D__iaa_1__iaa_1, R__iaa_1, iaa_1) = z

	(d__D__ARF_1__ARF_1, d__D__ARF_1__iaa_1, d__D__iaa_1__iaa_1, d__G__ARF_1__ARF_1, d__G__ARF_1__iaa_1, d__R, d__arf__all, d__iaa_1__x, d__iaa__all, k__D__ARF_1__ARF_1, k__D__ARF_1__iaa_1, k__D__iaa_1__iaa_1, k__G__ARF_1__ARF_1, k__G__ARF_1__ARF_1__R__iaa_1, k__G__ARF_1__iaa_1, k__G__ARF_1__iaa_1__R__iaa_1, k__G__R__iaa_1, k__RNA__IAA, k__arf__all, x) = p

	__0__x0 =D__ARF_1__ARF_1 .* k__G__ARF_1__ARF_1
	__0__x1 =1 ./ (-__0__x0 - d__G__ARF_1__ARF_1)
	__0__x2 =D__ARF_1__iaa_1 .* k__G__ARF_1__iaa_1
	__0__x3 =__0__x0 .* __0__x1 .* __0__x2 + __0__x2
	__0__x4 =__0__x3 ./ (-__0__x3 - d__G__ARF_1__iaa_1)
	(G__ARF_1__ARF_1, G__ARF_1__iaa_1) = [-__0__x1 .* (__0__x0 .* __0__x4 + __0__x0), -__0__x4]

	 G = -G__ARF_1__ARF_1 - G__ARF_1__iaa_1 + 1

	dz .= (-2 * ARF_1 .^ 2 .* k__D__ARF_1__ARF_1 - ARF_1 .* d__arf__all - ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 + 2 * D__ARF_1__ARF_1 .* d__D__ARF_1__ARF_1 + D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1 + k__arf__all, ARF_1 .^ 2 .* k__D__ARF_1__ARF_1 - D__ARF_1__ARF_1 .* d__D__ARF_1__ARF_1, ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 - D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1, -D__iaa_1__iaa_1 .* d__D__iaa_1__iaa_1 + iaa_1 .^ 2 .* k__D__iaa_1__iaa_1, G .* k__G__R__iaa_1 + G__ARF_1__ARF_1 .* k__G__ARF_1__ARF_1__R__iaa_1 + G__ARF_1__iaa_1 .* k__G__ARF_1__iaa_1__R__iaa_1 - R__iaa_1 .* d__R, -ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 + D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1 + 2 * D__iaa_1__iaa_1 .* d__D__iaa_1__iaa_1 + R__iaa_1 .* k__RNA__IAA - d__iaa_1__x .* iaa_1 .* x - d__iaa__all .* iaa_1 - 2 * iaa_1 .^ 2 .* k__D__iaa_1__iaa_1)

end