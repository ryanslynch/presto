project('PRESTO_python', 'c',
  version: '5.2.0',
  license: 'GPL-2.0',
  default_options: [
    'buildtype=release',
  ],
)
#
# Other possibilities
#    'buildtype=debugoptimized',
#    'buildtype=debug',

cc = meson.get_compiler('c')

# Compile and prepare all PRESTO code and python packages
py_mod = import('python')
py3 = py_mod.find_installation(pure: false)
py3_dep = py3.dependency()

# Check the python headers
cc.check_header('Python.h', dependencies: [py3_dep], required: true)
glib = dependency('glib-2.0')
fftw = dependency('fftw3f') # Note the 'f' for the single-precision version!
gsl = dependency('gsl')
fits = dependency('cfitsio')
x11 = dependency('x11')
png = dependency('libpng')

libm = cc.find_library('m', required: false)
libpresto = cc.find_library('presto', required: true)
# PGPLOT: prefer pkg-config, fall back to a bare library search, and make
# sure cpgplot.h is actually reachable (it is in <includedir>/pgplot for
# conda-forge's pgplot package, and in $PGPLOT_DIR for classic source
# installs).  This mirrors the identical logic in the top-level meson.build.
pgplot = dependency('pgplot', required: false)
if not pgplot.found()
  pgplot = cc.find_library('pgplot', required: true)
endif
cpgplot = dependency('cpgplot', required: false)
if not cpgplot.found()
  cpgplot = cc.find_library('cpgplot', required: true)
endif
if not cc.has_header('cpgplot.h', dependencies: cpgplot)
  fs = import('fs')
  pg_hdr_cands = []
  if cpgplot.type_name() == 'pkgconfig'
    pg_hdr_cands += cpgplot.get_variable(pkgconfig: 'includedir') / 'pgplot'
  endif
  pg_env = run_command('sh', '-c', 'printf %s "$PGPLOT_DIR"', check: false).stdout()
  if pg_env != ''
    pg_hdr_cands += pg_env
  endif
  pg_hdr_found = false
  foreach dir : pg_hdr_cands
    if not pg_hdr_found and fs.exists(dir / 'cpgplot.h')
      cpgplot = declare_dependency(dependencies: cpgplot,
                                   compile_args: '-I' + dir)
      pg_hdr_found = true
    endif
  endforeach
  if not pg_hdr_found
    error('Cannot find cpgplot.h -- install PGPLOT dev files or set PGPLOT_DIR.')
  endif
endif

inc = include_directories('../include')

incdir_numpy = run_command(py3,
  [
    '-c',
    'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

subdir('ppgplot_src')
subdir('presto_src')
subdir('presto')
