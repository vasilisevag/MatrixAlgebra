#pragma once

#include <iostream>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <initializer_list>
#include <algorithm>

namespace mat{

    template <int32_t N>
    class IdentityMatrix;

    template <int32_t R, int32_t C>
    class ZeroMatrix;

    template <typename T>
    struct IntegerWrapper { 
        public:
            IntegerWrapper(T t) : t(t) {}
            template <typename R>
            T operator=(const R r){
                this->t = r;
                return this->t;
            }
            T& operator=(const IntegerWrapper& r){
                t = r.t;
                return *this;
            }
            T operator[](int32_t idx){
                return t;
            }
            operator T() const {return t;}
        private:
            T t;
    };

    template <typename T>
    struct IntegerWrapper2 {
        public:
            IntegerWrapper2(T t, std::remove_reference_t<T>* values) : t(t), values(values) {}
            template <typename R>
            T operator=(const R r){
                this->t = r;
                return this->t;
            }
            T& operator=(const IntegerWrapper2& r){
                t = r.t;
                return *this;
            }
            T operator[](int32_t idx){
                return values[idx];
            }
            operator std::remove_reference_t<T>&() const {return t;}
        private:
            T t;
            std::remove_reference_t<T>* values;
    };

    template <typename T, int32_t R, int32_t C>
    struct init_type{
        using type = std::initializer_list<std::initializer_list<T>>;
    };

    template <typename T, int32_t R>
    struct init_type<T, R, 1>{
        using type = std::initializer_list<T>;
    };

    template <typename T, int32_t C>
    struct init_type<T, 1, C>{
        using type = std::initializer_list<T>;
    };

    template <typename T, int32_t R, int32_t C>
    using init_type_t = typename init_type<T, R, C>::type;



    template <typename T, int32_t R, int32_t C>
    class DenseMatrix {
        public:
            DenseMatrix(const init_type_t<T, R, C>& init_list){
                if constexpr (C == 1 || R == 1)
                    std::copy(init_list.begin(), init_list.end(), values);
                else {
                    for(auto i = init_list.begin(); i != init_list.end(); i++)
                        std::copy(i->begin(), i->end(), &values[(i - init_list.begin())*C]);
                }
            }
            DenseMatrix() = default;
            DenseMatrix(const DenseMatrix&) = default;
            DenseMatrix(DenseMatrix&&) = default;
            DenseMatrix& operator= (const DenseMatrix&) = default;
            DenseMatrix& operator= (DenseMatrix &&) = default;

            DenseMatrix operator-() const {
                DenseMatrix resultMatrix;
                for(int32_t rowIdx = 0; rowIdx < R; rowIdx++) 
                    for(int32_t colIdx = 0; colIdx < C; colIdx++)
                        resultMatrix[rowIdx][colIdx] = -values[rowIdx*C + colIdx];
                
                return resultMatrix;
            };
            constexpr int32_t rows() const {return R;}
            constexpr int32_t cols() const {return C;}
            decltype(auto) operator[](int32_t idx) {
                if constexpr (C == 1)
                    return IntegerWrapper<decltype(this->values[0])>(values[idx]);
                else if constexpr (R == 1){
                    return IntegerWrapper2<decltype(this->values[0])>(values[idx], values);
                    //T* p = values;
                    //return p;
                }
                else
                    return &values[idx*C];
            }
            decltype(auto) operator[](int32_t idx) const {
                if constexpr (C == 1)
                    return IntegerWrapper<decltype(this->values[0])>(values[idx]);
                else if constexpr (R == 1){
                    return IntegerWrapper2<decltype(this->values[0])>(values[idx], values);
                    //const T* p = values;
                    //return p;
                }
                else
                    return &values[idx*C];
            }
        private:
            T values[R*C] = {0};
    };

    template <int32_t N>
    class IdentityMatrix {
        public:
            DenseMatrix<int, N, N> operator-() const {
                DenseMatrix<int, N, N> resultMatrix;
                for(int32_t rowIdx = 0; rowIdx < N; rowIdx++) 
                    resultMatrix[rowIdx][rowIdx] = -1;
                
                return resultMatrix;
            };
            constexpr int32_t rows() const {return N;}
            constexpr int32_t cols() const {return N;}
        private:
    };

    template <int32_t R, int32_t C>
    class ZeroMatrix {
        public:
            ZeroMatrix operator-() const {
                return *this;
            };
            constexpr int32_t rows() const {return R;}
            constexpr int32_t cols() const {return C;}
        private:
    };

    template <typename T, int32_t N>
    using Vector = DenseMatrix<T, N, 1>;

    template <typename T>
    struct is_dense_matrix {constexpr static bool value = false;};

    template <typename T, int32_t R, int32_t C>
    struct is_dense_matrix<DenseMatrix<T, R, C>> {constexpr static bool value = true;};

    template <typename T>
    constexpr bool is_dense_matrix_v = is_dense_matrix<T>::value;

    template <typename T>
    struct is_identity_matrix {constexpr static bool value = false;};

    template <int32_t N>
    struct is_identity_matrix<IdentityMatrix<N>> {constexpr static bool value = true;};

    template <typename T>
    constexpr bool is_identity_matrix_v = is_identity_matrix<T>::value;

    template <typename T>
    struct is_zero_matrix {constexpr static bool value = false;};

    template <int32_t R, int32_t C>
    struct is_zero_matrix<ZeroMatrix<R, C>> {constexpr static bool value = true;};

    template <typename T>
    constexpr bool is_zero_matrix_v = is_zero_matrix<T>::value;

    template <typename T>
    T DenseSubtraction(const T& matrixLeft, const T& matrixRight) {
        T resultMatrix;
        for(int32_t rowIdx = 0; rowIdx < matrixLeft.rows(); rowIdx++)
            for(int32_t colIdx = 0; colIdx < matrixLeft.cols(); colIdx++)
                resultMatrix[rowIdx][colIdx] = matrixLeft[rowIdx][colIdx] - matrixRight[rowIdx][colIdx];

        return resultMatrix;
    }

    template <typename T>
    T DenseAddition(const T& matrixLeft, const T& matrixRight) {
        T resultMatrix;
        for(int32_t rowIdx = 0; rowIdx < matrixLeft.rows(); rowIdx++)
            for(int32_t colIdx = 0; colIdx < matrixLeft.cols(); colIdx++)
                resultMatrix[rowIdx][colIdx] = matrixLeft[rowIdx][colIdx] + matrixRight[rowIdx][colIdx];

        return resultMatrix;
    }

    template <template<typename, int32_t, int32_t> typename Matrix, typename T, int32_t RLEFT, typename U, int32_t CRIGHT, int32_t CCR>
    Matrix<std::common_type_t<T, U>, RLEFT, CRIGHT> DenseMultiplication(const Matrix<T, RLEFT, CCR>& matrixLeft, const Matrix<U, CCR, CRIGHT>& matrixRight) {
        
        Matrix<std::common_type_t<T, U>, RLEFT, CRIGHT> resultMatrix;

        for(int32_t rowIdx = 0; rowIdx < RLEFT; rowIdx++)
            for(int32_t colIdx = 0; colIdx < CRIGHT; colIdx++)
                for(int32_t k = 0; k < CCR; k++)
                    resultMatrix[rowIdx][colIdx] = resultMatrix[rowIdx][colIdx] + matrixLeft[rowIdx][k]*matrixRight[k][colIdx];

        return resultMatrix;
    }

    template <typename T>
    T DenseAdditionWithIdentity(const T& matrix){
        T resultMatrix;
        for(int32_t rowIdx = 0; rowIdx < matrix.rows(); rowIdx++)
            resultMatrix[rowIdx][rowIdx] = matrix[rowIdx][rowIdx] + 1;
        
        return resultMatrix;
    }

    template <typename T>
    T DenseSubtractionWithIdentity(const T& matrix){
        T resultMatrix;
        for(int32_t rowIdx = 0; rowIdx < matrix.rows(); rowIdx++)
            resultMatrix[rowIdx][rowIdx] = matrix[rowIdx][rowIdx] - 1;
        
        return resultMatrix;
    }

    template <typename MatrixLeft, typename MatrixRight>
    auto operator+(const MatrixLeft& matrixLeft, const MatrixRight& matrixRight) {
        static_assert(matrixLeft.cols() == matrixRight.cols() && matrixLeft.rows() == matrixRight.rows());

        if constexpr (is_dense_matrix_v<MatrixLeft> && is_dense_matrix_v<MatrixRight>){
            return DenseAddition(matrixLeft, matrixRight);
        }
        else if constexpr (is_identity_matrix_v<MatrixLeft>){
            return DenseAdditionWithIdentity(matrixRight);
        }
        else if constexpr (is_identity_matrix_v<MatrixRight>){
            return DenseAdditionWithIdentity(matrixLeft);
        }
        else if constexpr (is_zero_matrix_v<MatrixLeft>){
            return matrixRight;
        }
        else if constexpr (is_zero_matrix_v<MatrixRight>){
            return matrixLeft;
        }
    }

    template <typename MatrixLeft, typename MatrixRight>
    auto operator-(const MatrixLeft& matrixLeft, const MatrixRight& matrixRight) {
        static_assert(matrixLeft.cols() == matrixRight.cols() && matrixLeft.rows() == matrixRight.rows());

        if constexpr (is_dense_matrix_v<MatrixRight>){
            return DenseSubtraction(matrixLeft, matrixRight);
        }
        else if constexpr (is_identity_matrix_v<MatrixRight>){
            return DenseSubtractionWithIdentity(matrixLeft);
        }
        else if constexpr (is_zero_matrix_v<MatrixLeft>){
            return -matrixRight;
        }
        else if constexpr (is_zero_matrix_v<MatrixRight>){
            return matrixLeft;
        }
    }

    template <typename MatrixLeft, typename MatrixRight, typename = std::void_t<decltype(std::declval<MatrixLeft>().cols()), decltype(std::declval<MatrixLeft>().cols()), decltype(std::declval<MatrixLeft>().rows()), decltype(std::declval<MatrixLeft>().cols())>>
    auto operator*(const MatrixLeft& matrixLeft, const MatrixRight& matrixRight) {

        static_assert(matrixLeft.cols() == matrixRight.rows());

        if constexpr (is_dense_matrix_v<MatrixLeft> && is_dense_matrix_v<MatrixRight>){
            return DenseMultiplication(matrixLeft, matrixRight);
        }
        else if constexpr (is_identity_matrix_v<MatrixLeft>){
            return matrixRight;
        }
        else if constexpr (is_identity_matrix_v<MatrixRight>){
            return matrixLeft;
        }
        else if constexpr (is_zero_matrix_v<MatrixLeft> || is_zero_matrix_v<MatrixRight>){
            return ZeroMatrix<matrixLeft.rows(), matrixRight.cols()>();
        }
    }

    template <typename D, int32_t R, int32_t C>
    DenseMatrix<D, C, R> T(const DenseMatrix<D, R, C>& matrix){
        DenseMatrix<D, C, R> resultMatrix;
        for(int rowIdx = 0; rowIdx < R; rowIdx++)
            for(int colIdx = 0; colIdx < C; colIdx++)
                resultMatrix[colIdx][rowIdx] = matrix[rowIdx][colIdx];

        return resultMatrix;
    }

    template <typename Matrix, typename L>
    requires std::is_arithmetic_v<L>
    Matrix operator*(L lamba, const Matrix& matrix){
        Matrix resultMatrix;
        for(int rowIdx = 0; rowIdx < matrix.rows(); rowIdx++)
            for(int colIdx = 0; colIdx < matrix.cols(); colIdx++)
                resultMatrix[rowIdx][colIdx] = lamba*matrix[rowIdx][colIdx];

        return resultMatrix;
    }

    template <typename T, int32_t R, int32_t C>
    void printDense(const DenseMatrix<T, R, C>& matrix) {
        for(int32_t rowIdx = 0; rowIdx < R; rowIdx++) {
            for(int32_t colIdx = 0; colIdx < C; colIdx++)
                std::cout << matrix[rowIdx][colIdx] << ' ';
            std::cout << std::endl;
        }
    }

    template <int32_t N>
    void printIdentity(const IdentityMatrix<N>& matrix) {
        for(int32_t rowIdx = 0; rowIdx < N; rowIdx++) {
            for(int32_t colIdx = 0; colIdx < N; colIdx++){
                if(colIdx == rowIdx) std::cout << 1 << ' ';
                else
                    std::cout << 0 << ' ';
            }
            std::cout << std::endl;
        }
    }

    template <int32_t R, int32_t C>
    void printZero(const ZeroMatrix<R, C>& matrix) {
        for(int32_t rowIdx = 0; rowIdx < R; rowIdx++) {
            for(int32_t colIdx = 0; colIdx < C; colIdx++)
                std::cout << 0 << ' ';
            std::cout << std::endl;
        }
    }

    template <typename Matrix> requires (is_dense_matrix_v<Matrix> || is_identity_matrix_v<Matrix> || is_zero_matrix_v<Matrix>)
    std::ostream& operator<<(std::ostream& out, const Matrix& matrix){
        if constexpr (is_dense_matrix_v<Matrix>)
            printDense(matrix);
        else if constexpr (is_identity_matrix_v<Matrix>)
            printIdentity(matrix);
        else if constexpr (is_zero_matrix_v<Matrix>)
            printZero(matrix);
        
        return out;
    }
};
