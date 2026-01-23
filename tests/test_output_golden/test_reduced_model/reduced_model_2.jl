function reduced_auxin_signalling_pathway_f_deriv!(dz, z, p, t=0.0)
	(ARF_1, ARF_2, D__ARF_1__ARF_1, D__ARF_1__ARF_2, D__ARF_1__iaa_1, D__ARF_2__ARF_2, D__ARF_2__iaa_1, D__iaa_1__iaa_1, R__iaa_1, iaa_1) = z

	(d__D__ARF_1__ARF_1, d__D__ARF_1__ARF_2, d__D__ARF_1__iaa_1, d__D__ARF_2__ARF_2, d__D__ARF_2__iaa_1, d__D__iaa_1__iaa_1, d__G__ARF_1__ARF_1, d__G__ARF_1__ARF_2, d__G__ARF_1__iaa_1, d__G__ARF_2__ARF_2, d__G__ARF_2__iaa_1, d__R, d__arf__all, d__iaa_1__x, d__iaa__all, k__D__ARF_1__ARF_1, k__D__ARF_1__ARF_2, k__D__ARF_1__iaa_1, k__D__ARF_2__ARF_2, k__D__ARF_2__iaa_1, k__D__iaa_1__iaa_1, k__G__ARF_1__ARF_1, k__G__ARF_1__ARF_1__R__iaa_1, k__G__ARF_1__ARF_2, k__G__ARF_1__ARF_2__R__iaa_1, k__G__ARF_1__R__iaa_1, k__G__ARF_1__iaa_1, k__G__ARF_1__iaa_1__R__iaa_1, k__G__ARF_2__ARF_2, k__G__ARF_2__ARF_2__R__iaa_1, k__G__ARF_2__R__iaa_1, k__G__ARF_2__iaa_1, k__G__ARF_2__iaa_1__R__iaa_1, k__G__R__iaa_1, k__RNA__ARF, k__RNA__IAA, k__arf__all, x) = p

	x0 = D__ARF_1__ARF_1 .* k__G__ARF_1__ARF_1
	x1 = 1 ./ (-d__G__ARF_1__ARF_1 - x0)
	x2 = D__ARF_2__iaa_1 .* k__G__ARF_2__iaa_1
	x3 = x0 .* x1
	x4 = x2 .* x3
	x5 = D__ARF_1__ARF_2 .* k__G__ARF_1__ARF_2
	x6 = x3 .* x5
	x7 = x5 + x6
	x8 = 1 ./ (-d__G__ARF_1__ARF_2 - x7)
	x9 = x2 + x4
	x10 = -x8 .* x9
	x11 = D__ARF_1__iaa_1 .* k__G__ARF_1__iaa_1
	x12 = x11 .* x3
	x13 = x11 + x12
	x14 = -x13 .* x8
	x15 = x14 .* x7
	x16 = x13 - x15
	x17 = -x7
	x18 = x13 + x14 .* x17
	x19 = 1 ./ (-d__G__ARF_1__iaa_1 - x18)
	x20 = x10 .* x17 + x9
	x21 = -x19 .* x20
	x22 = D__ARF_2__ARF_2 .* k__G__ARF_2__ARF_2
	x23 = x22 .* x3
	x24 = x22 + x23
	x25 = -x24 .* x8
	x26 = x25 .* x7
	x27 = x17 .* x25 + x24
	x28 = -x19 .* x27
	x29 = x16 .* x28
	x30 = -x18
	x31 = x27 + x28 .* x30
	x32 = 1 ./ (-d__G__ARF_2__ARF_2 - x31)
	x33 = x20 + x21 .* x30
	x34 = -x32 .* x33
	x35 = -x31
	x36 = (-x10 .* x7 - x16 .* x21 + x2 - x34 .* (x24 - x26 - x29) + x4) ./ (-d__G__ARF_2__iaa_1 - x33 - x34 .* x35)
	x37 = x32 .* (x22 + x23 - x26 - x29 - x35 .* x36)
	x38 = x19 .* (x11 + x12 - x15 - x30 .* x36 - x30 .* x37)
	x39 = x8 .* (-x17 .* x36 - x17 .* x37 - x17 .* x38 + x5 + x6)
	(G__ARF_1__ARF_1, G__ARF_1__ARF_2, G__ARF_1__iaa_1, G__ARF_2__ARF_2, G__ARF_2__iaa_1) = [-x1 .* (x0 .* x36 + x0 .* x37 + x0 .* x38 + x0 .* x39 + x0), -x39, -x38, -x37, -x36]

	 G = 1 - (G__ARF_1__ARF_1 + G__ARF_1__ARF_2 + G__ARF_1__iaa_1 + G__ARF_2__ARF_2 + G__ARF_2__iaa_1)

	dz .= (-2 * ARF_1 .^ 2 .* k__D__ARF_1__ARF_1 - ARF_1 .* ARF_2 .* k__D__ARF_1__ARF_2 - ARF_1 .* d__arf__all - ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 + 2 * D__ARF_1__ARF_1 .* d__D__ARF_1__ARF_1 + D__ARF_1__ARF_2 .* d__D__ARF_1__ARF_2 + D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1 + k__arf__all, -ARF_1 .* ARF_2 .* k__D__ARF_1__ARF_2 - 2 * ARF_2 .^ 2 .* k__D__ARF_2__ARF_2 - ARF_2 .* d__arf__all - ARF_2 .* iaa_1 .* k__D__ARF_2__iaa_1 + D__ARF_1__ARF_2 .* d__D__ARF_1__ARF_2 + 2 * D__ARF_2__ARF_2 .* d__D__ARF_2__ARF_2 + D__ARF_2__iaa_1 .* d__D__ARF_2__iaa_1 + k__arf__all, ARF_1 .^ 2 .* k__D__ARF_1__ARF_1 - D__ARF_1__ARF_1 .* d__D__ARF_1__ARF_1, ARF_1 .* ARF_2 .* k__D__ARF_1__ARF_2 - D__ARF_1__ARF_2 .* d__D__ARF_1__ARF_2, ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 - D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1, ARF_2 .^ 2 .* k__D__ARF_2__ARF_2 - D__ARF_2__ARF_2 .* d__D__ARF_2__ARF_2, ARF_2 .* iaa_1 .* k__D__ARF_2__iaa_1 - D__ARF_2__iaa_1 .* d__D__ARF_2__iaa_1, -D__iaa_1__iaa_1 .* d__D__iaa_1__iaa_1 + iaa_1 .^ 2 .* k__D__iaa_1__iaa_1, G .* k__G__R__iaa_1 + G__ARF_1__ARF_1 .* k__G__ARF_1__ARF_1__R__iaa_1 + G__ARF_1__ARF_2 .* k__G__ARF_1__ARF_2__R__iaa_1 + G__ARF_1__iaa_1 .* k__G__ARF_1__iaa_1__R__iaa_1 + G__ARF_2__ARF_2 .* k__G__ARF_2__ARF_2__R__iaa_1 + G__ARF_2__iaa_1 .* k__G__ARF_2__iaa_1__R__iaa_1 - R__iaa_1 .* d__R, -ARF_1 .* iaa_1 .* k__D__ARF_1__iaa_1 - ARF_2 .* iaa_1 .* k__D__ARF_2__iaa_1 + D__ARF_1__iaa_1 .* d__D__ARF_1__iaa_1 + D__ARF_2__iaa_1 .* d__D__ARF_2__iaa_1 + 2 * D__iaa_1__iaa_1 .* d__D__iaa_1__iaa_1 + R__iaa_1 .* k__RNA__IAA - d__iaa_1__x .* iaa_1 .* x - d__iaa__all .* iaa_1 - 2 * iaa_1 .^ 2 .* k__D__iaa_1__iaa_1)

end