import copy
import random
from typing import Counter

import numpy as np
from qulacs.gate import CNOT, CZ, RX, H, Z


def hamil_think(jw_qulacs_hamiltonian):
    n_qubit = jw_qulacs_hamiltonian.get_qubit_count()
    # このようなパターン分けが可能だという前提のもと、分ける
    # XYが各2つ、 最小のqubitがX -> XX,YYの次数が2を発見
    # XXが1つ -> XX,YYの次数が1を発見
    # 構成を考える
    # Z の項
    # XX,YYの次数が1の項
    # ([0,1],-1) -> X0X1 + Y0Y1
    # ([0,1],5) -> X0X1Z5 + Y0Y1Y5
    # ([0,1],[4,7]) ->X0X1X4Z5Z6X7+X0X1Y4Z5Z6Y7+Y0Y1X4Z5Z6X7+Y0Y1Y4Z5Z6Y7
    all_Z = {}
    one_XX = {}
    two_XX = {}
    # (ビット、X=1 Y=2 Z=3) のtupleのtuple で表す
    # tupleの順序はXY -> Z で、　その中ではビット順
    # 項の間にあるZを除去した形式で表示
    for i in range(jw_qulacs_hamiltonian.get_term_count()):
        pau = jw_qulacs_hamiltonian.get_term(i)
        id_XYZ = zip(pau.get_index_list(), pau.get_pauli_id_list())
        pauli_ids = np.zeros(n_qubit, dtype=int)
        for aaa in id_XYZ:
            (index, pid) = aaa
            pauli_ids[index] = pid
        XYs = []
        Zs = []
        inZ = 0
        for j in range(n_qubit):
            if pauli_ids[j] == 1:  # X
                XYs.append((j, 1))
                inZ += 1
                inZ %= 2
            elif pauli_ids[j] == 2:  # Y
                XYs.append((j, 2))
                inZ += 1
                inZ %= 2
            elif pauli_ids[j] % 2 != inZ:  # Z
                Zs.append((j, 3))
        XYZt = tuple(XYs + Zs)
        if len(XYs) == 4:
            two_XX[XYZt] = (pau.get_coef(), 1.0, 0, 0)
        if len(XYs) == 2:
            one_XX[XYZt] = (pau.get_coef(), 1.0, 0, 0)
        if len(XYs) == 0:
            all_Z[XYZt] = (pau.get_coef(), 1.0, 0, 0)
    return (all_Z, one_XX, two_XX)


def make_pair_patan(n_qubit):
    ha_qubit = n_qubit // 2
    if ha_qubit % 2 == 0:
        loop_qubit = ha_qubit - 1
    else:
        loop_qubit = ha_qubit

    pata_XXU = []
    sya_hai = random.sample(list(range(0, ha_qubit)), ha_qubit)

    pata_Uyobi = []
    pata_Dyobi = []

    for i in range(loop_qubit):
        XX_pairs = []
        if ha_qubit % 2 == 0:
            a = i
            b = loop_qubit
            a = sya_hai[a]
            b = sya_hai[b]
            XX_pairs.append((min(a, b), max(a, b)))

        for j in range(1, (ha_qubit + 1) // 2):
            a = (j + i) % loop_qubit
            b = (loop_qubit - j + i) % loop_qubit
            a = sya_hai[a]
            b = sya_hai[b]
            XX_pairs.append((min(a, b), max(a, b)))

        if ha_qubit == 5 or ha_qubit == 6:
            (a, b) = XX_pairs[-1]
            (c, d) = XX_pairs[-2]
            yobi_pairs_A = [(min(a, c), max(a, c)), (min(b, d), max(b, d))]
            yobi_pairs_B = [(min(a, d), max(a, d)), (min(b, c), max(b, c))]
            if ha_qubit == 6:
                yobi_pairs_A.append(XX_pairs[0])
                yobi_pairs_B.append(XX_pairs[0])
            yobi_pairs_A.sort()
            yobi_pairs_B.sort()
            pata_Uyobi.append(yobi_pairs_A)
            pata_Uyobi.append(yobi_pairs_B)

        XX_pairs.sort()
        pata_XXU.append(XX_pairs)
    pata_XXD = []
    sya_hai = random.sample(list(range(0, ha_qubit)), ha_qubit)
    for i in range(loop_qubit):
        XX_pairs = []
        if ha_qubit % 2 == 0:
            a = i
            b = loop_qubit
            a = sya_hai[a] + ha_qubit
            b = sya_hai[b] + ha_qubit
            XX_pairs.append((min(a, b), max(a, b)))

        for j in range(1, (ha_qubit + 1) // 2):
            a = (j + i) % loop_qubit
            b = (loop_qubit - j + i) % loop_qubit
            a = sya_hai[a] + ha_qubit
            b = sya_hai[b] + ha_qubit
            XX_pairs.append((min(a, b), max(a, b)))

        if ha_qubit == 5 or ha_qubit == 6:
            (a, b) = XX_pairs[-1]
            (c, d) = XX_pairs[-2]
            yobi_pairs_A = [(min(a, c), max(a, c)), (min(b, d), max(b, d))]
            yobi_pairs_B = [(min(a, d), max(a, d)), (min(b, c), max(b, c))]
            if ha_qubit == 6:
                yobi_pairs_A.append(XX_pairs[0])
                yobi_pairs_B.append(XX_pairs[0])
            yobi_pairs_A.sort()
            yobi_pairs_B.sort()
            pata_Dyobi.append(yobi_pairs_A)
            pata_Dyobi.append(yobi_pairs_B)

        XX_pairs.sort()
        pata_XXD.append(XX_pairs)

    pata_yobi = []
    for i in range(len(pata_Uyobi)):
        pata_yobi.append(pata_Uyobi[i] + pata_Dyobi[i])

    return (pata_XXU, pata_XXD, pata_yobi)


def get_energy_EX(
    circuit, n_qubit, n_shots, state, hamiltonian_data, pata_data, executor
):
    (all_Z, one_XX, two_XX) = hamiltonian_data
    (pata_XXU, pata_XXD, pata_yobi) = pata_data
    ha_qubit = n_qubit // 2

    if ha_qubit % 2 == 0:
        loop_qubit = ha_qubit - 1
    else:
        loop_qubit = ha_qubit
    # リーグ戦の総当たりの方法
    # http://www.bea.hi-ho.ne.jp/ems-ontime/infotext1_8.html

    # 上でリーグ、下でリーグを行う

    # この、「ループリーグ」で取得した組を、さらにランダムな順列で動かす

    # 各pairにXX,他はZZで測定　(all_Zとone_XXとtwo_XXの一部)

    # two_XXの測定を行う
    pata_XXUDs = copy.copy(pata_yobi)
    for i in range(loop_qubit):
        for j in range(loop_qubit):
            # 「上下ヒット」があるかどうかを数え、ないならばcontinue
            hit_aru = False
            for pa_U in pata_XXU[i]:
                for pa_D in pata_XXD[j]:
                    (q, w) = pa_U
                    (e, r) = pa_D
                    if ((q, 1), (w, 1), (e, 1), (r, 1)) in two_XX:
                        hit_aru = True
                        break
            if not hit_aru:
                continue

            pata_XXUDs.append(pata_XXU[i] + pata_XXD[j])
    for pata_XXUD in pata_XXUDs:

        gen_cir = circuit.copy()
        pair_syo = list(range(n_qubit))

        for pa in pata_XXUD:
            (a, b) = pa
            pair_syo[a] = a
            pair_syo[b] = a

        for k in range(n_qubit):
            for h in range(k + 1, n_qubit):
                if pair_syo[k] > pair_syo[h]:
                    gen_cir.add_gate(CZ(k, h))
        # CZは、　2つの異なるpairに属するbitを見たとき、　bitの大小 != pairの小さい側の大小 の時に掛ける
        for k in range(n_qubit):
            gen_cir.add_gate(Z(k))
            gen_cir.add_gate(RX(k, np.pi / 2))

        for pa in pata_XXUD:
            (a, b) = pa
            gen_cir.add_gate(CNOT(a, b))
            gen_cir.add_gate(H(a))

        # 実測定を行う
        counts = Counter(
            executor.sampling(
                gen_cir,
                state_int=state,
                n_qubits=n_qubit,
                n_shots=n_shots,
            )
        )

        samples = []
        for sample, it_kaz in counts.items():
            rev_binary = np.binary_repr(sample).rjust(n_qubit, "0")
            sample = [1] * n_qubit
            for k in range(n_qubit):
                if rev_binary[n_qubit - 1 - k] == "1":
                    sample[k] = -1
            samples.append((sample, it_kaz))

        for k in range(len(pata_XXUD)):
            for h in range(k + 1, len(pata_XXUD)):
                (q, w) = pata_XXUD[k]
                (e, r) = pata_XXUD[h]

                if w < e:
                    ter_tuple = [
                        ((q, 1), (w, 1), (e, 1), (r, 1)),
                        ((q, 2), (w, 2), (e, 1), (r, 1)),
                        ((q, 1), (w, 1), (e, 2), (r, 2)),
                        ((q, 2), (w, 2), (e, 2), (r, 2)),
                    ]
                    XYZi_pry = [1, 1, 1, 1]
                elif e < w and w < r:
                    ter_tuple = [
                        ((q, 1), (e, 2), (w, 2), (r, 1)),
                        ((q, 2), (e, 2), (w, 1), (r, 1)),
                        ((q, 1), (e, 1), (w, 2), (r, 2)),
                        ((q, 2), (e, 1), (w, 1), (r, 2)),
                    ]
                    XYZi_pry = [1, -1, -1, 1]
                elif r < w:
                    ter_tuple = [
                        ((q, 1), (e, 2), (r, 2), (w, 1)),
                        ((q, 2), (e, 2), (r, 2), (w, 2)),
                        ((q, 1), (e, 1), (r, 1), (w, 1)),
                        ((q, 2), (e, 1), (r, 1), (w, 2)),
                    ]
                    XYZi_pry = [-1, -1, -1, -1]

                if ter_tuple[0] in two_XX:
                    (coef, cally, mes_sum, mes_count) = two_XX[ter_tuple[0]]
                    for sample, it_kaz in samples:
                        mes_sum += sample[q] * sample[e] * it_kaz * XYZi_pry[0]
                    two_XX[ter_tuple[0]] = (
                        coef,
                        cally,
                        mes_sum,
                        mes_count + n_shots,
                    )
                if ter_tuple[1] in two_XX:
                    (coef, cally, mes_sum, mes_count) = two_XX[ter_tuple[1]]
                    for sample, it_kaz in samples:
                        mes_sum += sample[w] * sample[e] * it_kaz * XYZi_pry[1]
                    two_XX[ter_tuple[1]] = (
                        coef,
                        cally,
                        mes_sum,
                        mes_count + n_shots,
                    )
                if ter_tuple[2] in two_XX:
                    (coef, cally, mes_sum, mes_count) = two_XX[ter_tuple[2]]
                    for sample, it_kaz in samples:
                        mes_sum += sample[q] * sample[r] * it_kaz * XYZi_pry[2]
                    two_XX[ter_tuple[2]] = (
                        coef,
                        cally,
                        mes_sum,
                        mes_count + n_shots,
                    )
                if ter_tuple[3] in two_XX:
                    (coef, cally, mes_sum, mes_count) = two_XX[ter_tuple[3]]
                    for sample, it_kaz in samples:
                        mes_sum += sample[w] * sample[r] * it_kaz * XYZi_pry[3]
                    two_XX[ter_tuple[3]] = (
                        coef,
                        cally,
                        mes_sum,
                        mes_count + n_shots,
                    )

                if ((q, 1), (w, 1)) in one_XX:
                    (coef, cally, mes_sum, mes_count) = one_XX[(q, 1), (w, 1)]
                    for sample, it_kaz in samples:
                        mes_sum += sample[q] * it_kaz
                    one_XX[(q, 1), (w, 1)] = (
                        coef,
                        cally,
                        mes_sum,
                        mes_count + n_shots,
                    )
                if ((q, 2), (w, 2)) in one_XX:
                    (coef, cally, mes_sum, mes_count) = one_XX[(q, 2), (w, 2)]
                    for sample, it_kaz in samples:
                        mes_sum += sample[w] * it_kaz
                    one_XX[(q, 2), (w, 2)] = (
                        coef,
                        cally,
                        mes_sum,
                        mes_count + n_shots,
                    )
                if ((e, 1), (r, 1)) in one_XX:
                    (coef, cally, mes_sum, mes_count) = one_XX[(e, 1), (r, 1)]
                    for sample, it_kaz in samples:
                        mes_sum += sample[e] * it_kaz
                    one_XX[(e, 1), (r, 1)] = (
                        coef,
                        cally,
                        mes_sum,
                        mes_count + n_shots,
                    )
                if ((e, 2), (r, 2)) in one_XX:
                    (coef, cally, mes_sum, mes_count) = one_XX[(e, 2), (r, 2)]
                    for sample, it_kaz in samples:
                        mes_sum += sample[r] * it_kaz
                    one_XX[(e, 2), (r, 2)] = (
                        coef,
                        cally,
                        mes_sum,
                        mes_count + n_shots,
                    )

    # one_XXとall_Zの測定を行う
    for i in range(n_qubit):
        for j in range(i + 1, n_qubit):
            exists = False
            for k in range(n_qubit):
                if i == k or j == k:
                    continue
                if ((i, 1), (j, 1), (k, 3)) in one_XX:
                    exists = True
                    break
            if not exists:
                continue

            gen_cir = circuit.copy()

            for k in range(i + 1, j):
                gen_cir.add_gate(CZ(k, j))
            gen_cir.add_gate(Z(i))
            gen_cir.add_gate(RX(i, np.pi / 2))
            gen_cir.add_gate(Z(j))
            gen_cir.add_gate(RX(j, np.pi / 2))
            gen_cir.add_gate(CNOT(i, j))
            gen_cir.add_gate(H(i))

            counts = Counter(
                executor.sampling(
                    gen_cir,
                    state_int=state,
                    n_qubits=n_qubit,
                    n_shots=n_shots,
                )
            )
            samples = []
            for sample, it_kaz in counts.items():
                rev_binary = np.binary_repr(sample).rjust(n_qubit, "0")
                sample = [1] * n_qubit
                for k in range(n_qubit):
                    if rev_binary[n_qubit - 1 - k] == "1":
                        sample[k] = -1
                samples.append((sample, it_kaz))

            if ((i, 1), (j, 1)) in one_XX:
                (coef, cally, mes_sum, mes_count) = one_XX[((i, 1), (j, 1))]
                for sample, it_kaz in samples:
                    mes_sum += sample[i] * it_kaz
                one_XX[((i, 1), (j, 1))] = (coef, cally, mes_sum, mes_count + n_shots)
            if ((i, 2), (j, 2)) in one_XX:
                (coef, cally, mes_sum, mes_count) = one_XX[((i, 2), (j, 2))]
                for sample, it_kaz in samples:
                    mes_sum += sample[j] * it_kaz
                one_XX[((i, 2), (j, 2))] = (coef, cally, mes_sum, mes_count + n_shots)

            for k in range(n_qubit):
                if i == k or j == k:
                    continue

                if ((i, 1), (j, 1), (k, 3)) in one_XX:
                    (coef, cally, mes_sum, mes_count) = one_XX[((i, 1), (j, 1), (k, 3))]
                    for sample, it_kaz in samples:
                        mes_sum += sample[i] * sample[k] * it_kaz
                    one_XX[((i, 1), (j, 1), (k, 3))] = (
                        coef,
                        cally,
                        mes_sum,
                        mes_count + n_shots,
                    )
                if ((i, 2), (j, 2), (k, 3)) in one_XX:
                    (coef, cally, mes_sum, mes_count) = one_XX[((i, 2), (j, 2), (k, 3))]
                    for sample, it_kaz in samples:
                        mes_sum += sample[j] * sample[k] * it_kaz
                    one_XX[((i, 2), (j, 2), (k, 3))] = (
                        coef,
                        cally,
                        mes_sum,
                        mes_count + n_shots,
                    )

                if ((k, 3),) in all_Z:
                    (coef, cally, mes_sum, mes_count) = all_Z[((k, 3),)]
                    for sample, it_kaz in samples:
                        mes_sum += sample[k] * it_kaz
                    all_Z[((k, 3),)] = (coef, cally, mes_sum, mes_count + n_shots)

                for h in range(k + 1, n_qubit):
                    if i == h or j == h:
                        continue
                    if ((k, 3), (h, 3)) in all_Z:
                        (coef, cally, mes_sum, mes_count) = all_Z[((k, 3), (h, 3))]
                        for sample, it_kaz in samples:
                            mes_sum += sample[k] * sample[h] * it_kaz
                        all_Z[((k, 3), (h, 3))] = (
                            coef,
                            cally,
                            mes_sum,
                            mes_count + n_shots,
                        )

    if () in all_Z:
        (coef, cally, mes_sum, mes_count) = all_Z[()]
        all_Z[()] = (coef, cally, 1, 1)

    ret = 0.0
    for it in all_Z:
        (coef, cally, mes_sum, mes_count) = all_Z[it]
        if mes_count >= 1:
            ret += coef * cally * mes_sum / mes_count
            mes_sum = 0
            mes_count = 0
            cally *= 0.7
        cally += 0.3
        all_Z[it] = (coef, cally, mes_sum, mes_count)
    for it in one_XX:
        (coef, cally, mes_sum, mes_count) = one_XX[it]
        if mes_count >= 1:
            ret += coef * cally * mes_sum / mes_count
            mes_sum = 0
            mes_count = 0
            cally *= 0.7
        cally += 0.3
        one_XX[it] = (coef, cally, mes_sum, mes_count)
    for it in two_XX:
        (coef, cally, mes_sum, mes_count) = two_XX[it]
        if mes_count >= 1:
            ret += coef * cally * mes_sum / mes_count
            mes_sum = 0
            mes_count = 0
            cally *= 0.7
        cally += 0.3
        two_XX[it] = (coef, cally, mes_sum, mes_count)
    return ret.real
