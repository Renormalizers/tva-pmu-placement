"""Convert a MATPOWER .m case into the hackathon's JSON grid format.
Zero-injection buses: no load (Pd = Qd = 0), no shunt, no in-service generator. Out-of-service
branches and isolated buses are dropped; parallel lines become one edge.
Usage: python matpower_import.py case.m out.json"""
import json, re, sys


def block(text, name):
    m = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", text, re.S)
    rows = []
    for line in m.group(1).splitlines():
        line = line.split("%")[0].strip().rstrip(";").strip()
        if line:
            rows.append([float(x) for x in line.replace(";", " ").split()])
    return rows


def convert(path, out):
    t = open(path).read()
    bus, gen, br = block(t, "bus"), block(t, "gen"), block(t, "branch")
    live = {int(b[0]) for b in bus if int(b[1]) != 4}
    gens = {int(g[0]) for g in gen if len(g) > 7 and g[7] > 0}
    lines = set()
    for b in br:
        f, to, status = int(b[0]), int(b[1]), b[10]
        if status > 0 and f in live and to in live and f != to:
            lines.add((min(f, to), max(f, to)))
    used = {x for l in lines for x in l}
    buses = sorted(used)
    load = {str(int(b[0])): round(b[2], 2) for b in bus if int(b[0]) in used}
    zib = sorted(int(b[0]) for b in bus if int(b[0]) in used and b[2] == 0 and b[3] == 0
                 and b[4] == 0 and b[5] == 0 and int(b[0]) not in gens)
    g = {"name": path.split("/")[-1].replace(".m", ""), "buses": buses, "lines": sorted(lines),
         "zero_injection": zib, "generator_buses": sorted(gens & used), "load_mw": load}
    json.dump(g, open(out, "w"))
    print(f"{g['name']}: {len(buses)} buses, {len(lines)} lines, {len(zib)} zero-injection, {len(g['generator_buses'])} generator buses")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
