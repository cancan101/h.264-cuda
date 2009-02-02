################################################################################
#
# Common build script
#
################################################################################

#.SUFFIXES : .cu .cu_dbg.o .c_dbg.o .cpp_dbg.o .cu_rel.o .c_rel.o .cpp_rel.o .cubin

# Add new SM Versions here as devices with new Compute Capability are released
SM_VERSIONS := sm_10 sm_11 sm_12 sm_13

CUDA_INSTALL_PATH ?= /usr/local/cuda

ifdef cuda-install
	CUDA_INSTALL_PATH := $(cuda-install)
endif

CUDA_SDK_PATH ?= /home/$(USER)/NVIDIA_CUDA_SDK

ifdef cuda-sdk
	CUDA_SDK_PATH := $(cuda-sdk)
endif

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?= 
#ROOTDIR    ?= .
ROOTBINDIR ?= .
#$(ROOTDIR)/bin
BINDIR     ?= $(ROOTBINDIR)
#/$(OSLOWER)
ROOTOBJDIR ?= .
#obj
LIBDIR     := $(CUDA_SDK_PATH)/lib
COMMONDIR  := $(CUDA_SDK_PATH)/common

# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 
CXX        := g++
CC         := gcc

# Includes
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include  -I$(COMMONDIR)/inc
#-I$(COMMONDIR)/inc

# architecture flag for cubin build
CUBIN_ARCH_FLAG := -m32

# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

# Compiler-specific flags
NVCCFLAGS := 
CUDACFLAGS    := $(CWARN_FLAGS)

# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g
	CFLAGS      += -g
	NVCCFLAGS   += -D_DEBUG -g
	BINSUBDIR   := .
	#debug
	LIBSUFFIX   := D	
else 
	##COMMONFLAGS += -O3 
	BINSUBDIR   := .
	#release
	LIBSUFFIX   :=
	##NVCCFLAGS   += --compiler-options -fno-strict-aliasing
	##CFLAGS      += -fno-strict-aliasing
endif

# append optional arch/SM version flags (such as -arch sm_11)
#NVCCFLAGS += $(SMVERSIONFLAGS)

# architecture flag for cubin build
CUBIN_ARCH_FLAG := -m32

# detect if 32 bit or 64 bit system
HP_64 =	$(shell uname -m | grep 64)

# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
ifeq ($(USEGLLIB),1)

	ifneq ($(DARWIN),)
		OPENGLLIB := -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL -lGLU $(COMMONDIR)/lib/$(OSLOWER)/libGLEW.a
	else
		OPENGLLIB := -lGL -lGLU -lX11 -lXi -lXmu

		ifeq "$(strip $(HP_64))" ""
			OPENGLLIB += -lGLEW -L/usr/X11R6/lib
		else
			OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
		endif
	endif

	CUBIN_ARCH_FLAG := -m64
endif

ifeq ($(USEGLUT),1)
	ifneq ($(DARWIN),)
		OPENGLLIB += -framework GLUT
	else
		OPENGLLIB += -lglut
	endif
endif

ifeq ($(USEPARAMGL),1)
	PARAMGLLIB := -lparamgl$(LIBSUFFIX)
endif

ifeq ($(USERENDERCHECKGL),1)
	RENDERCHECKGLLIB := -lrendercheckgl$(LIBSUFFIX)
endif


ifeq ($(USECUDPP), 1)
	ifeq "$(strip $(HP_64))" ""
		CUDPPLIB := -lcudpp
	else
		CUDPPLIB := -lcudpp64
	endif

	CUDPPLIB := $(CUDPPLIB)$(LIBSUFFIX)

	ifeq ($(emu), 1)
		CUDPPLIB := $(CUDPPLIB)_emu
	endif
endif

# Libs
LIB       := -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) 
CUDALFFLAGS := -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) 
ifeq ($(USEDRVAPI),1)
   LIB += -lcuda  $(CUDPPLIB)
   CUDALFFLAGS += -lcuda $(CUDPPLIB) ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB)
else
   LIB += -lcudart $(CUDPPLIB)
   CUDALFFLAGS += -lcudart $(CUDPPLIB) ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB)
endif

ifeq ($(USECUFFT),1)
  ifeq ($(emu),1)
    LIB += -lcufftemu
    CUDALFFLAGS += -lcufftemu
  else
    LIB += -lcufft
    CUDALFFLAGS += -lcufft
  endif
endif

ifeq ($(USECUBLAS),1)
  ifeq ($(emu),1)
    LIB += -lcublasemu
    CUDALFFLAGS += -lcublasemu
  else
    LIB += -lcublas
    CUDALFFLAGS += -lcublas
  endif
endif

# Lib/exe configuration
ifneq ($(STATIC_LIB),)
else
	LIB += -lcutil$(LIBSUFFIX)
	# Device emulation configuration
	ifeq ($(emu), 1)
		NVCCFLAGS   += -deviceemu
		CUDACCFLAGS += 
		#BINSUBDIR   := emu$(BINSUBDIR)
		# consistency, makes developing easier
		CFLAGS	+= -D__DEVICE_EMULATION__
	endif
endif

# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

################################################################################
# Check for input flags and set compiler flags appropriately
################################################################################
ifeq ($(fastmath), 1)
	NVCCFLAGS += -use_fast_math
endif

ifeq ($(keep), 1)
	NVCCFLAGS += -keep
	NVCC_KEEP_CLEAN := *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx
endif

ifdef maxregisters
	NVCCFLAGS += -maxrregcount $(maxregisters)
endif

# Add cudacc flags
NVCCFLAGS += $(CUDACCFLAGS)

# workaround for mac os x cuda 1.1 compiler issues
ifneq ($(DARWIN),)
	NVCCFLAGS += --host-compilation=C
endif

# Add common flags
NVCCFLAGS += --host-compilation=c
NVCCFLAGS += $(COMMONFLAGS)
CUDACFLAGS    += $(COMMONFLAGS)

ifeq ($(nvcc_warn_verbose),1)
	NVCCFLAGS += $(addprefix --compiler-options ,$(CXXWARN_FLAGS)) 
	NVCCFLAGS += --compiler-options -fno-strict-aliasing
endif

################################################################################
# Set up object files
################################################################################
OBJDIR := $(ROOTOBJDIR)
#$(ROOTOBJDIR)/$(BINSUBDIR)
#OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c.o,$(notdir $(CUDACFILES)))
CUDAOBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cuo, $(CUFILES))

################################################################################
# Set up cubin files
################################################################################
CUBINDIR := $(SRCDIR)data 
CUBINS +=  $(patsubst %.cu,$(CUBINDIR)/%.cubin,$(notdir $(CUBINFILES)))

################################################################################
# Rules
################################################################################
#$(OBJDIR)/%.c.o : $(SRCDIR)%.c $(C_DEPS)
#	$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cuo : $(SRCDIR)%.cu $(CU_DEPS)
	@$(VERBOSE)$(NVCC) $(NVCCFLAGS)  --no-align-double $(SMVERSIONFLAGS) -o $@ -c $<


$(CUBINDIR)/%.cubin : $(SRCDIR)%.cu cubindirectory
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -cubin $<

#
# The following definition is a template that gets instantiated for each SM
# version (sm_10, sm_13, etc.) stored in SMVERSIONS.  It does 2 things:
# 1. It adds to OBJS a .cu_sm_XX.o for each .cu file it finds in CUFILES_sm_XX.
# 2. It generates a rule for building .cu_sm_XX.o files from the corresponding 
#    .cu file.
#
# The intended use for this is to allow Makefiles that use common.mk to compile
# files to different Compute Capability targets (aka SM arch version).  To do
# so, in the Makefile, list files for each SM arch separately, like so:
#
# CUFILES_sm_10 := mycudakernel_sm10.cu app.cu
# CUFILES_sm_12 := anothercudakernel_sm12.cu
#
define SMVERSION_template
OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu_$(1).o,$(notdir $(CUFILES_$(1))))
$(OBJDIR)/%.cu_$(1).o : $(SRCDIR)%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) -o $$@ -c $$< $(NVCCFLAGS) -arch $(1)
endef

# This line invokes the above template for each arch version stored in
# SM_VERSIONS.  The call funtion invokes the template, and the eval
# function interprets it as make commands.
$(foreach smver,$(SM_VERSIONS),$(eval $(call SMVERSION_template,$(smver))))

cubindirectory:
	$(VERBOSE)mkdir -p $(CUBINDIR)

makedirectories:
	#$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)

tidy :
	$(VERBOSE)find . | egrep "#" | xargs rm -f
	$(VERBOSE)find . | egrep "\~" | xargs rm -f

cleancuda : tidy
	$(VERBOSE)rm -f $(CUDAOBJS)
	$(VERBOSE)rm -f $(CUBINS)
	$(VERBOSE)rm -f $(NVCC_KEEP_CLEAN)

clobber : clean
	$(VERBOSE)rm -rf $(ROOTOBJDIR)
