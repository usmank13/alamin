"""Host-only clock and description bindings for Rhizome checkouts predating them."""
from pathlib import Path


# Rhizome links its static archives into each extension. GCC's GNU-unique
# inline variable also shares one clock across RTLD_LOCAL Python extensions.
CLOCK = r"""
#include <atomic>
#include <cstdint>
namespace alamin {
__attribute__((visibility("default"))) inline std::atomic<int64_t> clock_ns{-1};
}
"""

HOOKS = {
    'src/time/robot_time.cpp': (
        'Time now() {',
        '''Time now() {
  auto ns = alamin::clock_ns.load(std::memory_order_relaxed);
  if (ns >= 0) return nanoseconds(ns);''',
        CLOCK,
    ),
    'src/pyzome/core.cpp': (
        '  m.def("now", &robot_time::now);',
        '''  m.def("now", &robot_time::now);
  m.def("set_sim_time", [](robot_time::Time t) {
    alamin::clock_ns.store(t.nanoseconds(), std::memory_order_relaxed);
  });''',
        CLOCK,
    ),
    'src/nav/bindings/nav.cpp': (
        'NB_MODULE(nav, m) {',
        '''NB_MODULE(nav, m) {
  m.def("set_test_description", [](const RobotDescriptionMsg &msg) {
    RobotDescription::set_test_description(msg);
  });''',
        '',
    ),
}


def host_overlay(root, output):
    """Generate only the three host translation units containing the test hooks."""
    root, output = Path(root), Path(output)
    directory = output/'sim-odometry'
    directory.mkdir(exist_ok=True)
    makefile = directory/'overlay.mk'
    rules = []
    native_clock = '"set_sim_time"' in (root/'src/pyzome/core.cpp').read_text()
    for relative, (anchor, replacement, prefix) in HOOKS.items():
        original = root/relative
        source = original.read_text()
        if prefix == CLOCK and native_clock:
            continue
        if relative.endswith('/nav.cpp') and '"set_test_description"' in source:
            continue
        if source.count(anchor) != 1:
            raise ValueError(f'Simulation host hook does not match {relative}')
        generated = directory/original.name
        generated.write_text(prefix + source.replace(anchor, replacement))
        rules.append(f'$(BUILD_DIR_HOST)/{relative}.o: {generated}\n'
                     '\t@mkdir -p $(@D)\n'
                     f'\t$(CXX_HOST) $(CFLAGS_HOST) $(CXXFLAGS_HOST) $(TGT_FLAGS) -I{original.parent} '
                     f'-c {generated} -MF $(@:.o=.d) -MT $@ -o $@\n')
    makefile.write_text(''.join(rules))
    return makefile
