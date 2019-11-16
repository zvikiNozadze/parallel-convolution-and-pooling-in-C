# parallel-convolution-and-pooling-in-C
 ### თუ სადმე შეცდომაა გამაგებინე და გამოვასწორებ.
 ###### ტესტებზე გასაშვებად "gcc convolve_tests.c" და მერე ./a.out, memory test-ზე "valgrind --leak-check=yes ./a.out"
 ###### struct ები აღწერილია "convolve_babee.h" ში
 
 ##### ამოცანა 1: maxpool_2d, convolution_2d -ს დაწერა
 
 ##### ამოცანა 2: maxpool_3d, convolution_3d - ს დაწერა
 
 ##### ამოცანა 3: maxpool_3d, convolution_3d - ის გაპარალელურება multythread რომ იყოს 3D ტენსორი რეალურად 2 D ტენსოების მასივია და დაწერე ისე რო 3D ტენზორში შემავალ, ყველა 2D ტენზორს ცალცალკე ითვლიდნენ thread ები და მერე, იმერჯებოდეს პასუხები. memory ს გამოყენება   ამ კურსში გვკიდია (reference :D )
 
 უნდა დაწერო 4 ფუნქცია ფაილში "convolve_babee.c" (maxpool_2d, convolution_2d, maxpool_3d, convolution_3d).
 ფუნქციები უნდა სრულებდნენ convolution და maxpool ოპერაციებს ტენსორებზე  (2d array-matrix, 3d-array-3dtensor).
  
    link ები დასახმარებლად (
      tensor      - https://en.wikipedia.org/wiki/Tensor
      convolution - https://www.youtube.com/watch?time_continue=16&v=43pm7yh-NYQ&feature=emb_logo
      maxpool     - https://www.youtube.com/watch?v=8oOgPUO-TBY
    ) 
 
 tensor - ჩვენთვის არის ნებისმიერი N განზომილებიანი მასივი. 
                  Scalar is a single number.\
                  Vector is an array of numbers.\
                  Matrix is a 2-D array of numbers.\
                  Tensors are N-D arrays of numbers.\
 
 convolution - არის ოპერაცია (ავღნიშნოთ @) ტენსორებზე A@b=c სადაც c-ს თითოეული ელემენტი შედგება A-ს ქვემატრიცზე(B-ს განზომილების) B მატრიცის ელემენტ-ელემენტ გადამრავლებით და მიღებულის აჯამვით (ქვემოთ მაგალითია :)).
 
 maxpool - არის მაქსიმალური ელემენტების ამორჩევა, A-ს (გადმოცემული განზომილების) ქვემატრიცებისგან
 
 მაგალითები :
  
    maxpool
    A = [[1, 2, 3],
         [0, 0, 5],
         [9, 2, 6]]
    განზომილებები: სიმაღლე = სიგანე = 2, stride (ანუ ნაბიჯის ზომა ანუ თითო განზომილების მიმართ რამდენით ვცვლით ადფილმდებარეობას) = 1
    პასუხი იქნება [[a, b
                    c, d]] 
    სადაც a = მაქსიმალურს A-ს ქვემატრიცაში [[1, 2    b = მაქსიმალურ A-ს ქვემატრ [[2, 3    c = max([[0, 0     d = [[0, 5
                                             0, 0]]                               0, 5]]            9, 2]])        2, 6]]
    ანუ პასუხია [[2, 5
                 9, 6]]
  სამ განზომილებაზე იგივეა განზოგადებული. დიდ A კუბში დავთვლით მთლიან შიგა კუბებიში მაქსიმალურებს.
  ## 2d ზე maxpool 
   დიდი მარჯვენა მატრიცა არის A (განზომილებებით 5 და 5). და ნელნელა მარცხენას ვაშენებთ პასუხს. ნაბიჯის ზომა ამ გიფშიც არის ერთი. და ფანჯრის ზომა რასაც ვატარებთ A ზე არის 3 და 3
    ![Image description](https://miro.medium.com/max/936/1*Fw-ehcNBR9byHtho-Rxbtw.gif)
      
    convolution იგივე A მატრიცზე B მატრიცით [1, -1, 0], იქნება C მატრიცა [[a,
                                                                           b,
                                                                           c]]
    სადაც a = ჯამი ელემენტების ([1,2,3] (ელემენტ-ელემენტ გამრავლება) [1,-1,0]) = ჯამი ელემენტების ([1, -2, 0]) = -1
    b = ჯამი ელემენტების ([0,0,5] (ელემენტ-ელემენტ გამრავლება) [1,-1,0]) = ჯამი ელემენტების ([0, 0, 0]) = 0
    c = ჯამი ელემენტების ([9,2,6] (ელემენტ-ელემენტ გამრავლება) [1,-1,0]) = ჯამი ელემენტების ([9, -2, 0]) = 7
    convolution სამ განზომილებაზე იგივეა განზოგადებული :) 
  
  ## 2d ზე convolution 
  ლურჯი არის A მატრიცა / ტენზორი
  
  მონაცრისფრო მატრიცა რაც ლურჯზე ტარდება არის B (რომ აწერია ქვედა მარჯვენა კუთხეებში ეგაა ელემენტები)
        თუ ვერ ხედავთ B = [[0,1,2], [2,2,0], [0,1,2]]  
  
  მწვანე (მარჯვენა) არის C
    
  ![Image description](https://miro.medium.com/max/428/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)
    
  ## ეს 3 D ზე convolution
   ![Image description](https://miro.medium.com/max/1288/1*wUVVgZnzBwYKgQyTBK_5sg.png)
