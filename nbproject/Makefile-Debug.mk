#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=nvcc
CCC=nvcc
CXX=nvcc
FC=gfortran
AS=as

# Macros
CND_PLATFORM=CUDA-Linux-x86
CND_DLIB_EXT=so
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/common/2d_malloc.o \
	${OBJECTDIR}/common/2d_malloc_cuda.o \
	${OBJECTDIR}/common/file_io.o \
	${OBJECTDIR}/common/filter_cuda.o \
	${OBJECTDIR}/main.o \
	${OBJECTDIR}/main_cuda.o


# C Compiler Flags
CFLAGS=-ccbin g++ -m64 -gencode arch=compute_11,code=sm_11

# CC Compiler Flags
CCFLAGS=-ccbin g++ -m64 -gencode arch=compute_11,code=sm_11
CXXFLAGS=-ccbin g++ -m64 -gencode arch=compute_11,code=sm_11

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-lm

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/cuda_conv

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/cuda_conv: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/cuda_conv ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/common/2d_malloc.o: common/2d_malloc.cpp 
	${MKDIR} -p ${OBJECTDIR}/common
	$(COMPILE.cc) -g -I/usr/include -I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/4.8/include -I/usr/local/cuda/include -o ${OBJECTDIR}/common/2d_malloc.o common/2d_malloc.cpp

${OBJECTDIR}/common/2d_malloc_cuda.o: common/2d_malloc_cuda.cu 
	${MKDIR} -p ${OBJECTDIR}/common
	$(COMPILE.cc) -g -I/usr/include -I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/4.8/include -I/usr/local/cuda/include -o ${OBJECTDIR}/common/2d_malloc_cuda.o common/2d_malloc_cuda.cu

${OBJECTDIR}/common/file_io.o: common/file_io.cpp 
	${MKDIR} -p ${OBJECTDIR}/common
	$(COMPILE.cc) -g -I/usr/include -I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/4.8/include -I/usr/local/cuda/include -o ${OBJECTDIR}/common/file_io.o common/file_io.cpp

${OBJECTDIR}/common/filter_cuda.o: common/filter_cuda.cu 
	${MKDIR} -p ${OBJECTDIR}/common
	$(COMPILE.cc) -g -I/usr/include -I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/4.8/include -I/usr/local/cuda/include -o ${OBJECTDIR}/common/filter_cuda.o common/filter_cuda.cu

${OBJECTDIR}/main.o: main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	$(COMPILE.cc) -g -I/usr/include -I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/4.8/include -I/usr/local/cuda/include -o ${OBJECTDIR}/main.o main.cpp

${OBJECTDIR}/main_cuda.o: main_cuda.cpp 
	${MKDIR} -p ${OBJECTDIR}
	$(COMPILE.cc) -g -I/usr/include -I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/4.8/include -I/usr/local/cuda/include -o ${OBJECTDIR}/main_cuda.o main_cuda.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/cuda_conv

# Subprojects
.clean-subprojects:
