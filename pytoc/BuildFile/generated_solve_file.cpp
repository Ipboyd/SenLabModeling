#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/bool.hpp>
#include <pythonic/types/int.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pythonic\include\types\float64.hpp>
#include <pythonic\include\types\ndarray.hpp>
#include <pythonic\types\float64.hpp>
#include <pythonic\types\ndarray.hpp>
namespace 
{
  namespace __pythran_generated_solve_file
  {
    struct solve_run
    {
      typedef void callable;
      typedef void pure;
      template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
      struct type
      {
        typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type0;
        typedef typename pythonic::returnable<__type0>::type __type1;
        typedef __type1 result_type;
      }  
      ;
      template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
      inline
      typename type<argument_type0, argument_type1, argument_type2>::result_type operator()(argument_type0 on_input, argument_type1 off_input, argument_type2 noise_token) const
      ;
    }  ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
    inline
    typename solve_run::type<argument_type0, argument_type1, argument_type2>::result_type solve_run::operator()(argument_type0 on_input, argument_type1 off_input, argument_type2 noise_token) const
    {
      return on_input;
    }
  }
}
#include <pythonic/python/exception_handler.hpp>
#ifdef ENABLE_PYTHON_MODULE
inline
typename __pythran_generated_solve_file::solve_run::type<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>::result_type solve_run0(pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>&& on_input, pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>&& off_input, pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>&& noise_token) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_generated_solve_file::solve_run()(on_input, off_input, noise_token);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}

static PyObject *
__pythran_wrap_solve_run0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    
    char const* keywords[] = {"on_input", "off_input", "noise_token",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>(args_obj[2]))
        return to_python(solve_run0(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long,long>>>(args_obj[2])));
    else {
        return nullptr;
    }
}

            static PyObject *
            __pythran_wrapall_solve_run(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_solve_run0(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "solve_run", "\n""    - solve_run(float64[:,:,:], float64[:,:,:], float64[:,:,:])", args, kw);
                });
            }


static PyMethodDef Methods[] = {
    {
    "solve_run",
    (PyCFunction)__pythran_wrapall_solve_run,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n""\n""    - solve_run(float64[:,:,:], float64[:,:,:], float64[:,:,:])"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "generated_solve_file",            /* m_name */
  "",         /* m_doc */
  -1,                  /* m_size */
  Methods,             /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL,                /* m_free */
};
PyMODINIT_FUNC
PyInit_generated_solve_file(void)
#ifndef _WIN32
__attribute__ ((visibility("default")))
#if defined(GNUC) && !defined(__clang__)
__attribute__ ((externally_visible))
#endif
#endif
;
PyMODINIT_FUNC
PyInit_generated_solve_file(void) {
    import_array();

    PyObject* theModule = PyModule_Create(&moduledef);
    if(! theModule)
        return theModule;

                #ifdef Py_GIL_DISABLED
                    PyUnstable_Module_SetGIL(theModule, Py_MOD_GIL_NOT_USED);
                #endif
    PyObject * theDoc = Py_BuildValue("(ss)",
                                      "0.18.0",
                                      "54a48961f53f5ad6b858f99be3ba415afa850f0c16673974c4f11a6b1b6c7450");
    if(! theDoc)
        return theModule;
    PyModule_AddObject(theModule,
                       "__pythran__",
                       theDoc);


    return theModule;
}

#endif