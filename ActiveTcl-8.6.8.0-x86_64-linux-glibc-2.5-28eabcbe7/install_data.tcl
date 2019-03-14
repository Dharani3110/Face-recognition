# -*- tcl -*-
# Copyright (C) 2001-2008 ActiveState Software Inc. Sep 28 2018
#
# Information for the installer which is dependent on the chosen
# distribution. Original source of the information is the script
# "build/setvars.sh".
#
# ActiveState ActiveTcl 8.6.8.0 Linux/x86_64, Fri Sep 28 19:15:17 MDT 2018
#

global    AT
array set AT {
    build       {28eabcbe7}
    NAME	{ActiveTcl}
    PGROUP	{ActiveTcl}
    dPGROUP	{ActiveTcl}
    VERSION	8.6.8
    MATURITY	final
    Company	ActiveState
    url,about	https://www.activestate.com/activetcl
    url,update	https://www.activestate.com/activetcl
    url,help	https://www.activestate.com/
    MODE	normal
    DISTC	at
    HELP	{ActiveTclHelp.chm}
    BLD_PFXPATH  {/home/gecko/work/tmp/ActiveState-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}
    BLD_EPFXPATH {/home/gecko/work/tmp/ActiveState-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}
    DEBUG        0
    THREADED     1
    tcl_VERSION  8.6
    ARCH         x86_64-linux-glibc-2.5
    TDK_ATVERS   8.6.8.0
    true_MODE	 normal
}

if {[string equal $AT(MODE) normal]} {
    if {$AT(THREADED)} {set tSuffix "-thread" } else {set tSuffix ""}
    set AT(LOG) [file join \
	lib ppm log [string tolower $AT(PGROUP)]$AT(tcl_VERSION)$tSuffix install.log]
    unset tSuffix
} else {
    set AT(LOG) [file join \
	lib ppm log [string tolower $AT(PGROUP)] install.log]
}

# When this file is used by the uninstaller SCRIPT_DIR is not defined.

if {[info exists ::SCRIPT_DIR]} {
    set _welcome [file join $::SCRIPT_DIR install_welcome.txt]

    global WELCOME
    set    WELCOME [read [set fh [open $_welcome r]]][close $fh]
    unset _welcome
}

if {[string equal $::AT(MODE) direct]} {
    # For the purposes of the installation the 'direct' distribution
    # is the same as the 'normal' ActiveTcl one.
    set ::AT(MODE) normal
}

# ========================================
set ::AT(LICENSE)      license-at8.6-thread.terms
set ::AT(README)       README-8.6-thread.txt
set ::AT(MANIFEST)     MANIFEST_at8.6.txt
set ::AT(tcl_revision) unknown
set ::AT(tk_revision)  unknown
set sofiles {}
set shfiles {}
lappend sofiles lib/libtcl8.6.so
lappend shfiles lib/tkimgConfig.sh
lappend shfiles lib/tifftclConfig.sh
lappend shfiles lib/TclxmlConfig.sh
lappend sofiles lib/libtclstub8.6.a
lappend shfiles lib/tclooConfig.sh
lappend shfiles lib/pngtclConfig.sh
lappend sofiles lib/libtkimgtiff1.4.6.so
lappend shfiles lib/nsfConfig.sh
lappend sofiles lib/libtifftclstub3.9.7.a
lappend sofiles lib/libtkstub8.6.a
lappend sofiles lib/libnsf2.1.0.so
lappend shfiles lib/zlibtclConfig.sh
lappend sofiles lib/libmk4.a
lappend sofiles lib/libtcl8.6.a
lappend shfiles lib/jpegtclConfig.sh
lappend sofiles lib/libpngtcl1.6.21.so
lappend sofiles lib/libtifftcl3.9.7.so
lappend sofiles lib/libtk8.6.so
lappend sofiles lib/libtkimgpng1.4.6.so
lappend sofiles lib/libsnackstub2.2.a
lappend shfiles lib/tkbltConfig.sh
lappend sofiles lib/libtk8.6.a
lappend shfiles lib/tclConfig.sh
lappend shfiles lib/tkConfig.sh
lappend sofiles lib/libmk4.so
lappend sofiles lib/libpngtclstub1.6.21.a
lappend sofiles lib/Mk4tcl/Mk4tcl.a
lappend sofiles lib/Mk4tcl/Mk4tcl.so
lappend sofiles lib/tcltls1.7.16/tcltls.so
lappend sofiles lib/thread2.8.2/libthread2.8.2.so
lappend sofiles lib/vfs1.4.2/libvfs1.4.2.a
lappend sofiles lib/vfs1.4.2/libvfs1.4.2.so
lappend sofiles lib/itk4.1.0/libitk4.1.0.so
lappend sofiles lib/tclx8.4/libtclx8.4.so
lappend sofiles lib/tcllibc/linux-x86_64/tcllibc.so
lappend sofiles lib/expect5.45.3/libexpect5.45.3.so
lappend sofiles lib/tkblt3.2/libtkblt3.2.so
lappend sofiles lib/tkblt3.2/libtkbltstub3.2.a
lappend sofiles lib/tdbc1.0.6/libtdbc1.0.6.so
lappend sofiles lib/tdbc1.0.6/libtdbcstub1.0.6.a
lappend shfiles lib/tdbc1.0.6/tdbcConfig.sh
lappend sofiles lib/tdbcpostgres1.0.6/libtdbcpostgres1.0.6.so
lappend sofiles lib/Tclxml3.2/libTclxml3.2.so
lappend sofiles lib/Tclxml3.2/libTclxmlstub3.2.a
lappend sofiles lib/trofs0.4.9/libtrofs0.4.9.so
lappend sofiles lib/tdbcmysql1.0.6/libtdbcmysql1.0.6.so
lappend sofiles lib/tcl8.6/critcl_md5c0.12/linux-x86_64/md5c.so
lappend sofiles lib/tbcload1.7/libtbcload1.7.so
lappend sofiles lib/snack2.2/libsound.so
lappend sofiles lib/snack2.2/libsnack.so
lappend sofiles lib/Tktable2.11/libTktable2.11.so
lappend sofiles lib/nsf2.1.0/libnsf2.1.0.so
lappend sofiles lib/nsf2.1.0/libnsfstub2.1.0.a
lappend sofiles lib/treectrl2.4.1/libtreectrl2.4.so
lappend sofiles lib/tdom0.8.3/libtdom0.8.3.so
lappend sofiles lib/tdom0.8.3/libtdomstub0.8.3.a
lappend sofiles lib/Trf2.1.4/libTrfstub2.1.4.a
lappend sofiles lib/Trf2.1.4/libTrf2.1.4.so
lappend sofiles lib/itcl4.1.1/libitcl4.1.1.so
lappend sofiles lib/itcl4.1.1/libitclstub4.1.1.a
lappend sofiles lib/itcl4.1.1/libitcl4.1.1.a
lappend shfiles lib/itcl4.1.1/itclConfig.sh
lappend sofiles lib/sqlite3.20.1/libsqlite3.20.1.so
lappend sofiles lib/tdbcodbc1.0.6/libtdbcodbc1.0.6.so
lappend sofiles lib/Img1.4.6/libtkimgjpeg1.4.6.so
lappend sofiles lib/Img1.4.6/libjpegtclstub9.2.a
lappend sofiles lib/Img1.4.6/libtkimgdted1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgtiff1.4.6.so
lappend sofiles lib/Img1.4.6/libtifftclstub3.9.7.a
lappend sofiles lib/Img1.4.6/libtkimg1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgps1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgpcx1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgsun1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgppm1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgico1.4.6.so
lappend sofiles lib/Img1.4.6/libzlibtclstub1.2.8.1.a
lappend sofiles lib/Img1.4.6/libpngtcl1.6.21.so
lappend sofiles lib/Img1.4.6/libtifftcl3.9.7.so
lappend sofiles lib/Img1.4.6/libtkimgwindow1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgbmp1.4.6.so
lappend sofiles lib/Img1.4.6/libzlibtcl1.2.8.1.so
lappend sofiles lib/Img1.4.6/libtkimgxpm1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgpng1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgpixmap1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimggif1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgsgi1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgxbm1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgstub1.4.6.a
lappend sofiles lib/Img1.4.6/libtkimgraw1.4.6.so
lappend sofiles lib/Img1.4.6/libtkimgtga1.4.6.so
lappend sofiles lib/Img1.4.6/libjpegtcl9.2.so
lappend sofiles lib/Img1.4.6/libpngtclstub1.6.21.a
