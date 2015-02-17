Paste this in a file called ".theanorc.txt" ('.txt' optional) at the C:/<User> level

[blas]

ldflags = -LC:\Users\Clayton\AppData\Local\Enthought\Canopy\App\appdata\canopy-1.5.2.2785.win-x86_64\Scripts -lmk2_core -lmk2_intel_thread -lmk2_rt

[gcc]

cxxflags = -IC:\MinGW\include
